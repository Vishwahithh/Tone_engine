#!/usr/bin/env python3
"""
Self-Training Tone Memory System
Analyzes your transcripts and builds a cumulative tone profile
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from anthropic import Anthropic

# Optional Supabase DB helper (works with both module/script import styles)
try:
    from .db import DB  # type: ignore
except Exception:  # pragma: no cover
    try:
        from db import DB  # type: ignore
    except Exception:
        DB = None  # type: ignore


class ToneEngine:
    def __init__(self, base_dir: str = "tone_engine", client_name: str = "default"):
        # Per-client local storage: tone_engine/clients/<client_name>/{transcripts,processed,profiles}
        self.client_name = client_name
        root_dir = Path(base_dir) / "clients" / client_name
        self.base_dir = root_dir
        self.transcripts_dir = self.base_dir / "transcripts"
        self.processed_dir = self.base_dir / "processed"
        self.profiles_dir = self.base_dir / "profiles"
        
        # Create directories
        for dir_path in [self.transcripts_dir, self.processed_dir, self.profiles_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Claude client
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        else:
            self.anthropic_client = None
            print("⚠️  No Anthropic API key found. Using enhanced fallback analysis.")
        
        self.tone_profile_path = self.profiles_dir / "tone_profile.json"

        # Optional DB integration
        self.db = None
        self.client_id = None
        if 'DB' in globals() and DB is not None:  # type: ignore
            try:
                self.db = DB()  # type: ignore
                if getattr(self.db, 'enabled')():
                    self.client_id = getattr(self.db, 'upsert_client')(client_name)
            except Exception as e:
                print(f"⚠️  DB init failed: {e}")
        
    def chunk_transcript(self, transcript: str, chunk_size: int = 7) -> List[str]:
        """Split transcript into meaningful chunks of ~5-10 sentences"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+\s+', transcript.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def analyze_tone(self, chunk: str) -> Dict[str, Any]:
        """Analyze tone using Claude API with fallback"""
        if not self.anthropic_client:
            return self._fallback_analysis(chunk)
            
        prompt = f"""
        Analyze the tone, style, and voice patterns in this text chunk. Return a JSON response with these fields:

        {{
            "tone": "primary emotional tone (e.g., reflective, energetic, analytical)",
            "style_markers": ["specific writing patterns", "word choices", "sentence structures"],
            "voice_qualities": ["conversational", "technical", "storytelling", etc.],
            "key_phrases": ["memorable phrases or expressions"],
            "structure_pattern": "how ideas are organized (linear, layered, circular, etc.)",
            "intensity": 7,
            "complexity": "simple/moderate/complex for idea depth"
        }}

        Text chunk:
        "{chunk}"

        Return only valid JSON, no other text.
        """
        
        try:
            # Try Claude models in order of preference
            models_to_try = [
                "claude-3-5-sonnet-20241022",  # Latest Sonnet
                "claude-3-5-sonnet-20240620",  # Previous Sonnet
                "claude-3-5-haiku-20241022",   # Latest Haiku
                "claude-3-haiku-20240307"      # Fallback Haiku
            ]
            
            for model in models_to_try:
                try:
                    response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=500,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    # Parse JSON from response
                    json_str = response.content[0].text.strip()
                    # Clean up any markdown formatting
                    if json_str.startswith("```json"):
                        json_str = json_str.replace("```json", "").replace("```", "").strip()
                    
                    result = json.loads(json_str)
                    print(f"✅ Using Claude model: {model}")
                    return result
                    
                except Exception as model_error:
                    print(f"⚠️  Model {model} failed: {str(model_error)[:100]}...")
                    continue
            
            # If all Claude models fail, use enhanced fallback
            print("❌ All Claude models failed, using enhanced fallback analysis")
            return self._fallback_analysis(chunk)
            
        except Exception as e:
            print(f"Error analyzing tone with Claude: {e}")
            return self._fallback_analysis(chunk)
    

    
    def _fallback_analysis(self, chunk: str) -> Dict[str, Any]:
        """Enhanced fallback analysis if API fails"""
        # Basic text analysis
        words = chunk.lower().split()
        sentences = chunk.count('.') + chunk.count('!') + chunk.count('?')
        
        # Simple tone detection based on keywords
        tone = "neutral"
        intensity = 5
        
        # Emotional indicators
        if any(word in chunk.lower() for word in ['excited', 'amazing', 'love', 'fantastic', 'incredible']):
            tone = "enthusiastic"
            intensity = 8
        elif any(word in chunk.lower() for word in ['think', 'analyze', 'consider', 'evaluate', 'examine']):
            tone = "analytical" 
            intensity = 6
        elif any(word in chunk.lower() for word in ['feel', 'believe', 'sense', 'wonder', 'ponder']):
            tone = "reflective"
            intensity = 6
        elif any(word in chunk.lower() for word in ['however', 'but', 'challenge', 'problem', 'difficult']):
            tone = "critical"
            intensity = 7
        
        # Complexity based on sentence length and vocabulary
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 3
        complexity = "simple" if avg_word_length < 4 else "moderate" if avg_word_length < 6 else "complex"
        
        # Style markers based on text patterns
        style_markers = []
        if "..." in chunk:
            style_markers.append("elliptical phrasing")
        if chunk.count(',') > sentences:
            style_markers.append("complex sentence structure")
        if any(chunk.startswith(phrase) for phrase in ["I think", "I feel", "I believe", "It seems"]):
            style_markers.append("personal perspective")
        if "?" in chunk:
            style_markers.append("questioning approach")
        
        if not style_markers:
            style_markers = ["standard phrasing"]
            
        return {
            "tone": tone,
            "style_markers": style_markers,
            "voice_qualities": ["conversational"],
            "key_phrases": [phrase for phrase in chunk.split('.')[0:2] if len(phrase.strip()) > 10],
            "structure_pattern": "linear",
            "intensity": intensity,
            "complexity": complexity
        }
    
    def process_transcript(self, transcript_file: str) -> List[Dict]:
        """Process a transcript file and analyze all chunks"""
        transcript_path = self.transcripts_dir / transcript_file
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        chunks = self.chunk_transcript(transcript)
        processed_chunks = []
        
        print(f"Processing {len(chunks)} chunks from {transcript_file}...")

        # Ensure transcript row in DB if available
        transcript_id = None
        if self.client_id and self.db and getattr(self.db, 'enabled')():
            try:
                transcript_id = getattr(self.db, 'ensure_transcript')(self.client_id, transcript_file)
            except Exception as e:
                print(f"⚠️  DB transcript upsert failed: {e}")
        
        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}...")
            
            tone_data = self.analyze_tone(chunk)
            
            chunk_data = {
                "chunk_id": f"{transcript_file.split('.')[0]}_chunk_{i+1:03d}",
                "source_file": transcript_file,
                "timestamp": datetime.now().isoformat(),
                "chunk_text": chunk,
                "analysis": tone_data
            }
            
            processed_chunks.append(chunk_data)
            
            # Save individual chunk
            chunk_file = self.processed_dir / f"{chunk_data['chunk_id']}.json"
            if chunk_file.exists():
                print(f"⚠️  Skipping duplicate chunk file: {chunk_file.name}")
                continue  # Skip this chunk
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=2, ensure_ascii=False)

            # Mirror to DB
            if transcript_id and self.db and getattr(self.db, 'enabled')():
                try:
                    getattr(self.db, 'upsert_chunk')(transcript_id, chunk_data)
                except Exception as e:
                    print(f"⚠️  DB chunk upsert failed: {e}")
        
        return processed_chunks
    
    def update_tone_profile(self, new_chunks: List[Dict]) -> Dict:
        """Update the cumulative tone profile with new chunks"""
        # Load existing profile
        if self.tone_profile_path.exists():
            with open(self.tone_profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
        else:
            profile = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "total_chunks": 0,
                "tone_frequencies": {},
                "style_patterns": {},
                "voice_evolution": [],
                "recent_examples": [],
                "dominant_traits": {}
            }
        
        # Update with new chunks
        for chunk in new_chunks:
            analysis = chunk['analysis']
            
            # Update tone frequencies
            tone = analysis.get('tone', 'neutral')
            profile['tone_frequencies'][tone] = profile['tone_frequencies'].get(tone, 0) + 1
            
            # Collect style markers
            for marker in analysis.get('style_markers', []):
                profile['style_patterns'][marker] = profile['style_patterns'].get(marker, 0) + 1
            
            # Add to recent examples (keep last 20)
            profile['recent_examples'].append({
                "chunk_id": chunk['chunk_id'],
                "tone": tone,
                "sample_text": chunk['chunk_text'][:100] + "...",
                "timestamp": chunk['timestamp']
            })
            
            if len(profile['recent_examples']) > 20:
                profile['recent_examples'] = profile['recent_examples'][-20:]
        
        # Update metadata
        profile['total_chunks'] += len(new_chunks)
        profile['last_updated'] = datetime.now().isoformat()
        
        # Calculate dominant traits
        if profile['tone_frequencies']:
            dominant_tone = max(profile['tone_frequencies'], key=profile['tone_frequencies'].get)
            profile['dominant_traits']['primary_tone'] = dominant_tone
        
        if profile['style_patterns']:
            top_patterns = sorted(profile['style_patterns'].items(), key=lambda x: x[1], reverse=True)[:5]
            profile['dominant_traits']['top_style_patterns'] = [pattern[0] for pattern in top_patterns]
        
        # Save updated profile
        with open(self.tone_profile_path, 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)

        # Mirror profile to DB
        if self.client_id and self.db and getattr(self.db, 'enabled')():
            try:
                getattr(self.db, 'upsert_tone_profile')(self.client_id, profile)
            except Exception as e:
                print(f"⚠️  DB profile upsert failed: {e}")
        
        return profile
    
    def generate_with_tone(self, prompt: str) -> str:
        """Generate content using your tone profile"""
        profile = None
        if self.tone_profile_path.exists():
            with open(self.tone_profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
        else:
            # Fallback: try DB if available
            if self.client_id and self.db and getattr(self.db, 'enabled')():
                try:
                    res = self.db.client.table("tone_profiles").select("profile").eq("client_id", self.client_id).single().execute()
                    if res.data and res.data.get("profile"):
                        profile = res.data["profile"]
                        # Persist locally for future reads
                        with open(self.tone_profile_path, 'w', encoding='utf-8') as f:
                            json.dump(profile, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
        if not profile:
            return "No tone profile found. Process some transcripts first!"
        
        tone_context = f"""
        Write in this voice profile:
        
        Primary Tone: {profile['dominant_traits'].get('primary_tone', 'conversational')}
        Top Style Patterns: {', '.join(profile['dominant_traits'].get('top_style_patterns', []))}
        
        Recent Voice Examples:
        {chr(10).join([ex['sample_text'] for ex in profile.get('recent_examples', [])[-3:]])}
        
        Task: {prompt}
        
        Write in that exact voice and style.
        """
        
        if not self.anthropic_client:
            return "❌ No Claude API key available. Add ANTHROPIC_API_KEY to your .env file to generate content."
        
        try:
            # Try Claude models for generation
            models_to_try = [
                "claude-3-5-sonnet-20241022",
                "claude-3-5-sonnet-20240620", 
                "claude-3-5-haiku-20241022",
                "claude-3-haiku-20240307"
            ]
            
            for model in models_to_try:
                try:
                    response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=1000,
                        messages=[{"role": "user", "content": tone_context}]
                    )
                    return response.content[0].text
                except Exception as model_error:
                    print(f"⚠️  Claude model {model} failed, trying next...")
                    continue
            
            return "❌ All Claude models failed. Your tone profile is ready, but generation is temporarily unavailable."
                
        except Exception as e:
            return f"Error generating content: {e}"
    
    def rephrase_in_my_tone(self, text: str) -> str:
        """Rephrase the given text in your own tone using your tone profile.
        Produces a concise rewrite that preserves meaning, avoids filler,
        and stays close to the original length.
        """
        # Load profile (same logic as generate_with_tone)
        profile = None
        if self.tone_profile_path.exists():
            with open(self.tone_profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
        else:
            if self.client_id and self.db and getattr(self.db, 'enabled')():
                try:
                    res = self.db.client.table("tone_profiles").select("profile").eq("client_id", self.client_id).single().execute()
                    if res.data and res.data.get("profile"):
                        profile = res.data["profile"]
                        with open(self.tone_profile_path, 'w', encoding='utf-8') as f:
                            json.dump(profile, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass

        if not profile:
            # Fall back to a minimal, safe return if no profile is present
            return text

        if not self.anthropic_client:
            return text

        # Build a strict rephrase instruction using the profile
        exemplars = "\n".join([ex.get('sample_text', '') for ex in profile.get('recent_examples', [])[-3:]])
        voice_summary = f"""
Primary Tone: {profile.get('dominant_traits', {}).get('primary_tone', 'conversational')}
Top Style Patterns: {', '.join(profile.get('dominant_traits', {}).get('top_style_patterns', []))}
Recent Exemplars:
{exemplars}
"""

        # Target token/length bounds (heuristic)
        words = text.split()
        approx_tokens = max(60, min(400, int(len(words) * 1.5) or 60))

        prompt = f"""
You are a precise style-preserving rewriter.

VOICE PROFILE
{voice_summary}

TASK
Rewrite the INPUT in the client's own voice.

Constraints:
- Preserve exact meaning and intent. Do NOT add new facts, small talk, or explanations.
- Keep length within ±20% of the input length.
- Avoid filler and hedging. Prefer clear, direct sentences.
- Maintain the same level of formality.
- Output only the rewritten text. No quotes, no preface.

INPUT
{text}
"""

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=approx_tokens,
                temperature=0.15,
                system="You are a precise, faithful editor that rewrites text in the user's own voice without adding content.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            return f"Error rephrasing: {e}"
    
    def get_tone_summary(self) -> str:
        """Get a summary of your current tone profile"""
        if not self.tone_profile_path.exists():
            return "No tone profile found. Process some transcripts first!"
        
        with open(self.tone_profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        summary = f"""
        📊 Your Voice Profile Summary
        
        Total Chunks Analyzed: {profile['total_chunks']}
        Last Updated: {profile.get('last_updated', 'Unknown')}
        
        🎯 Dominant Tone: {profile['dominant_traits'].get('primary_tone', 'Unknown')}
        
        📈 Tone Distribution:
        {chr(10).join([f"  {tone}: {count}" for tone, count in sorted(profile['tone_frequencies'].items(), key=lambda x: x[1], reverse=True)[:5]])}
        
        ✨ Top Style Patterns:
        {chr(10).join([f"  • {pattern}" for pattern in profile['dominant_traits'].get('top_style_patterns', [])[:5]])}
        """
        
        return summary


def main():
    """Main execution function"""
    print("🧠 Tone Engine - Claude-Powered Voice System")
    print("=" * 50)
    
    # Initialize engine
    engine = ToneEngine()
    
    # Check for Anthropic API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️  No Anthropic API key found in .env file")
        print("💡 Add ANTHROPIC_API_KEY=your_key_here to your .env file for best results")
        print("🔄 Continuing with enhanced fallback analysis...")
    
    # Example workflow
    print("\n📂 Available transcripts:")
    transcript_files = list(engine.transcripts_dir.glob("*.txt"))
    
    if not transcript_files:
        print("No transcript files found. Add some to /transcripts/")
        print("Example: Save your transcript as 'transcripts/2024-07-01.txt'")
        return
    
    for i, file in enumerate(transcript_files, 1):
        print(f"  {i}. {file.name}")
    
    # Process newest transcript
    latest_transcript = max(transcript_files, key=os.path.getctime)
    print(f"\n🔄 Processing latest transcript: {latest_transcript.name}")
    
    try:
        chunks = engine.process_transcript(latest_transcript.name)
        
        print(f"✅ Processed {len(chunks)} chunks")
        
        # Update tone profile
        print("📊 Updating tone profile...")
        profile = engine.update_tone_profile(chunks)
        print("🎯 Tone profile updated!")
        print(engine.get_tone_summary())
        
        # Example generation (only if API available)
        if engine.anthropic_client:
            print("\n💬 Example: Generating content in your voice...")
            example_prompt = "Write a short insight about how technology affects human connection."
            generated = engine.generate_with_tone(example_prompt)
            print(f"\nGenerated:\n{generated}")
        else:
            print("\n💡 Add ANTHROPIC_API_KEY to generate content in your voice!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()