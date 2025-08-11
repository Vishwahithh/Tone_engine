#!/usr/bin/env python3
"""
Advanced Tone Engine Features
- Voice evolution tracking
- Multi-context tone switching
- Batch processing
- Export/import profiles
"""

import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
from main import ToneEngine

class AdvancedToneEngine(ToneEngine):
    """Extended tone engine with advanced features"""
    
    def __init__(self, base_dir: str = "tone_engine", client_name: str = "default"):
        super().__init__(base_dir, client_name)
        self.evolution_file = self.profiles_dir / "voice_evolution.json"
        self.contexts_file = self.profiles_dir / "context_profiles.json"
    
    def track_voice_evolution(self, chunks: List[Dict], context: str = "default"):
        """Track how your voice changes over time"""
        evolution_data = self._load_evolution_data()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        if today not in evolution_data:
            evolution_data[today] = {}
        
        if context not in evolution_data[today]:
            evolution_data[today][context] = {
                "tone_distribution": {},
                "avg_intensity": 0,
                "complexity_score": 0,
                "chunk_count": 0
            }
        
        day_data = evolution_data[today][context]
        
        # Aggregate today's data
        intensities = []
        complexities = []
        
        for chunk in chunks:
            analysis = chunk['analysis']
            
            # Track tone distribution
            tone = analysis.get('tone', 'neutral')
            day_data['tone_distribution'][tone] = day_data['tone_distribution'].get(tone, 0) + 1
            
            # Track intensity and complexity
            intensity = analysis.get('intensity', 5)
            complexity = analysis.get('complexity', 'moderate')
            
            intensities.append(intensity)
            
            # Convert complexity to score
            complexity_map = {'simple': 1, 'moderate': 2, 'complex': 3}
            complexities.append(complexity_map.get(complexity, 2))
        
        # Update averages
        day_data['avg_intensity'] = np.mean(intensities) if intensities else 5
        day_data['complexity_score'] = np.mean(complexities) if complexities else 2
        day_data['chunk_count'] += len(chunks)
        
        self._save_evolution_data(evolution_data)
        return evolution_data
    
    def _load_evolution_data(self) -> Dict:
        """Load voice evolution data"""
        if self.evolution_file.exists():
            with open(self.evolution_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_evolution_data(self, data: Dict):
        """Save voice evolution data"""
        with open(self.evolution_file, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_context_profile(self, context_name: str, transcript_files: List[str]):
        """Create a tone profile for a specific context (work, personal, creative, etc.)"""
        context_chunks = []
        
        for file in transcript_files:
            chunks = self.process_transcript(file, use_claude=True)
            context_chunks.extend(chunks)
        
        # Create context-specific profile
        context_profile = self._build_context_profile(context_chunks, context_name)
        
        # Save context profile
        contexts = self._load_context_profiles()
        contexts[context_name] = context_profile
        self._save_context_profiles(contexts)
        
        return context_profile
    
    def _build_context_profile(self, chunks: List[Dict], context: str) -> Dict:
        """Build a context-specific tone profile"""
        profile = {
            "context": context,
            "created": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "tone_frequencies": {},
            "avg_intensity": 0,
            "dominant_patterns": [],
            "key_phrases": [],
            "sample_texts": []
        }
        
        intensities = []
        all_phrases = []
        
        for chunk in chunks:
            analysis = chunk['analysis']
            
            # Tone frequencies
            tone = analysis.get('tone', 'neutral')
            profile['tone_frequencies'][tone] = profile['tone_frequencies'].get(tone, 0) + 1
            
            # Collect intensities
            intensities.append(analysis.get('intensity', 5))
            
            # Collect key phrases
            phrases = analysis.get('key_phrases', [])
            all_phrases.extend(phrases)
            
            # Sample texts (keep best examples)
            if len(profile['sample_texts']) < 5:
                profile['sample_texts'].append({
                    "text": chunk['chunk_text'][:200] + "...",
                    "tone": tone,
                    "intensity": analysis.get('intensity', 5)
                })
        
        # Calculate averages
        profile['avg_intensity'] = np.mean(intensities) if intensities else 5
        
        # Find most common phrases
        phrase_counts = {}
        for phrase in all_phrases:
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        profile['key_phrases'] = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return profile
    
    def _load_context_profiles(self) -> Dict:
        """Load context profiles"""
        if self.contexts_file.exists():
            with open(self.contexts_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_context_profiles(self, contexts: Dict):
        """Save context profiles"""
        with open(self.contexts_file, 'w') as f:
            json.dump(contexts, f, indent=2, ensure_ascii=False)
    
    def generate_with_context(self, prompt: str, context: str = "default") -> str:
        """Generate content using a specific context profile"""
        contexts = self._load_context_profiles()
        
        if context not in contexts:
            return f"Context '{context}' not found. Available: {list(contexts.keys())}"
        
        profile = contexts[context]
        
        # Build context-aware prompt
        dominant_tone = max(profile['tone_frequencies'], key=profile['tone_frequencies'].get)
        top_phrases = [phrase[0] for phrase in profile['key_phrases'][:3]]
        
        context_prompt = f"""
        Write in this specific context voice:
        
        Context: {context}
        Dominant Tone: {dominant_tone}
        Average Intensity: {profile['avg_intensity']:.1f}/10
        Key Phrases to Echo: {', '.join(top_phrases)}
        
        Sample Voice:
        {profile['sample_texts'][0]['text'] if profile['sample_texts'] else 'No samples available'}
        
        Task: {prompt}
        
        Match this exact voice and context.
        """
        
        return self._generate_content(context_prompt)
    
    def _generate_content(self, prompt: str, temperature: float = 0.3) -> str:
        if not self.anthropic_client:
            return "No Anthropic API key available."
        try:
            resp = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=temperature,
                system="You are a careful ghostwriter matching the client’s exact voice and constraints.",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            return f"Error: {e}"
    
    def plot_voice_evolution(self, days: int = 30, context: str = "default"):
        """Plot how your voice has evolved over time"""
        evolution_data = self._load_evolution_data()
        
        # Get last N days of data
        end_date = datetime.now()
        dates = []
        intensities = []
        complexities = []
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            if date_str in evolution_data and context in evolution_data[date_str]:
                day_data = evolution_data[date_str][context]
                dates.append(date)
                intensities.append(day_data['avg_intensity'])
                complexities.append(day_data['complexity_score'])
        
        if not dates:
            print(f"No evolution data found for context '{context}'")
            return
        
        # Reverse to show chronological order
        dates.reverse()
        intensities.reverse()
        complexities.reverse()
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Intensity over time
        ax1.plot(dates, intensities, marker='o', linewidth=2, markersize=6)
        ax1.set_title(f'Voice Intensity Evolution - {context.title()} Context')
        ax1.set_ylabel('Intensity (1-10)')
        ax1.grid(True, alpha=0.3)
        
        # Complexity over time
        ax2.plot(dates, complexities, marker='s', color='orange', linewidth=2, markersize=6)
        ax2.set_title('Complexity Evolution')
        ax2.set_ylabel('Complexity Score (1-3)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        # Save plot
        plot_file = self.profiles_dir / f"voice_evolution_{context}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Evolution plot saved: {plot_file}")
        
        plt.show()
    
    def export_profile(self, filename: str = None) -> str:
        """Export your complete tone profile"""
        if not filename:
            filename = f"tone_profile_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "exported": datetime.now().isoformat(),
            "version": "2.0",
            "main_profile": {},
            "evolution_data": {},
            "context_profiles": {}
        }
        
        # Load all data
        if self.tone_profile_path.exists():
            with open(self.tone_profile_path, 'r') as f:
                export_data["main_profile"] = json.load(f)
        
        export_data["evolution_data"] = self._load_evolution_data()
        export_data["context_profiles"] = self._load_context_profiles()
        
        # Save export
        export_path = self.profiles_dir / filename
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(export_path)
    
    def import_profile(self, filepath: str):
        """Import a tone profile from export file"""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        # Import main profile
        if import_data.get("main_profile"):
            with open(self.tone_profile_path, 'w') as f:
                json.dump(import_data["main_profile"], f, indent=2, ensure_ascii=False)
        
        # Import evolution data
        if import_data.get("evolution_data"):
            self._save_evolution_data(import_data["evolution_data"])
        
        # Import context profiles
        if import_data.get("context_profiles"):
            self._save_context_profiles(import_data["context_profiles"])
        
        print(f"✅ Profile imported from {filepath}")
    
    def batch_process_transcripts(self, pattern: str = "*.txt", context: str = "default"):
        """Process multiple transcript files at once"""
        transcript_files = list(self.transcripts_dir.glob(pattern))
        
        if not transcript_files:
            print(f"No files found matching pattern: {pattern}")
            return
        
        print(f"🔄 Batch processing {len(transcript_files)} files...")
        
        all_chunks = []
        for file in transcript_files:
            print(f"Processing {file.name}...")
            try:
                chunks = self.process_transcript(file.name)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        if all_chunks:
            # Update main profile
            self.update_tone_profile(all_chunks)
            
            # Track evolution
            self.track_voice_evolution(all_chunks, context)
            
            print(f"✅ Batch processing complete: {len(all_chunks)} chunks processed")
        else:
            print("❌ No chunks processed")
    
    def analyze_tone_trends(self, days: int = 7) -> Dict:
        """Analyze tone trends over recent days"""
        evolution_data = self._load_evolution_data()
        
        recent_data = {}
        end_date = datetime.now()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            if date_str in evolution_data:
                recent_data[date_str] = evolution_data[date_str]
        
        if not recent_data:
            return {"error": "No recent data available"}
        
        # Analyze trends
        trends = {
            "date_range": f"{(end_date - timedelta(days=days-1)).strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            "total_days": len(recent_data),
            "contexts_analyzed": set(),
            "tone_evolution": {},
            "intensity_trend": [],
            "most_active_day": None,
            "dominant_tones": {}
        }
        
        daily_chunks = {}
        all_tones = {}
        
        for date, day_data in recent_data.items():
            daily_total = 0
            
            for context, context_data in day_data.items():
                trends["contexts_analyzed"].add(context)
                
                # Track chunk counts
                chunk_count = context_data.get('chunk_count', 0)
                daily_total += chunk_count
                
                # Track intensity
                intensity = context_data.get('avg_intensity', 5)
                trends["intensity_trend"].append({
                    "date": date,
                    "context": context,
                    "intensity": intensity
                })
                
                # Track tones
                for tone, count in context_data.get('tone_distribution', {}).items():
                    all_tones[tone] = all_tones.get(tone, 0) + count
            
            daily_chunks[date] = daily_total
        
        # Find most active day
        if daily_chunks:
            trends["most_active_day"] = max(daily_chunks, key=daily_chunks.get)
        
        # Convert sets to lists for JSON serialization
        trends["contexts_analyzed"] = list(trends["contexts_analyzed"])
        
        # Find dominant tones
        if all_tones:
            sorted_tones = sorted(all_tones.items(), key=lambda x: x[1], reverse=True)
            trends["dominant_tones"] = dict(sorted_tones[:5])
        
        return trends
    
    def get_voice_recommendations(self) -> List[str]:
        """Get recommendations for improving voice consistency"""
        recommendations = []
        
        # Analyze main profile
        if self.tone_profile_path.exists():
            with open(self.tone_profile_path, 'r') as f:
                profile = json.load(f)
            
            total_chunks = profile.get('total_chunks', 0)
            
            if total_chunks < 10:
                recommendations.append("📝 Process more transcripts (aim for 20+ chunks) for better voice modeling")
            
            # Check tone diversity
            tone_freq = profile.get('tone_frequencies', {})
            if len(tone_freq) == 1:
                recommendations.append("🎭 Try varying your speaking contexts for more tone diversity")
            
            # Check recent activity
            recent_examples = profile.get('recent_examples', [])
            if len(recent_examples) < 5:
                recommendations.append("🔄 Add more recent transcripts to keep your voice model current")
        
        # Analyze evolution data
        evolution_data = self._load_evolution_data()
        recent_dates = [d for d in evolution_data.keys() if datetime.now() - datetime.strptime(d, '%Y-%m-%d') <= timedelta(days=7)]
        
        if len(recent_dates) < 3:
            recommendations.append("📅 Try to process transcripts more regularly (3+ times per week)")
        
        # Check context diversity
        contexts = self._load_context_profiles()
        if len(contexts) < 2:
            recommendations.append("🏷️ Create different context profiles (work, personal, creative) for better voice switching")
        
        if not recommendations:
            recommendations.append("🎉 Your voice model looks great! Keep adding regular transcripts to maintain quality")
        
        return recommendations

def advanced_cli():
    """Advanced CLI with all features"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    engine = AdvancedToneEngine()
    
    print("🧠 Advanced Tone Engine CLI")
    print("Commands: process, batch, context, generate, evolve, trends, export, import, recommend, quit")
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command == "quit":
            break
            
        elif command == "process":
            files = list(engine.transcripts_dir.glob("*.txt"))
            if not files:
                print("No transcript files found")
                continue
            
            print("Available files:")
            for i, f in enumerate(files, 1):
                print(f"  {i}. {f.name}")
            
            try:
                choice = int(input("Choose file: ")) - 1
                if 0 <= choice < len(files):
                    context = input("Context (default/work/personal/creative): ").strip() or "default"
                    chunks = engine.process_transcript(files[choice].name)
                    engine.update_tone_profile(chunks)
                    engine.track_voice_evolution(chunks, context)
                    print(f"✅ Processed {len(chunks)} chunks in context '{context}'")
            except (ValueError, IndexError):
                print("Invalid choice")
        
        elif command == "batch":
            pattern = input("File pattern (*.txt): ").strip() or "*.txt"
            context = input("Context (default): ").strip() or "default"
            engine.batch_process_transcripts(pattern, context)
        
        elif command == "context":
            contexts = engine._load_context_profiles()
            print(f"Available contexts: {list(contexts.keys())}")
            
            action = input("Create new context? (y/n): ").strip().lower()
            if action == 'y':
                name = input("Context name: ").strip()
                files = input("Transcript files (comma-separated): ").strip().split(',')
                files = [f.strip() for f in files]
                
                try:
                    profile = engine.create_context_profile(name, files)
                    print(f"✅ Created context '{name}' with {profile['chunk_count']} chunks")
                except Exception as e:
                    print(f"Error: {e}")
        
        elif command == "generate":
            contexts = engine._load_context_profiles()
            if contexts:
                print(f"Available contexts: {list(contexts.keys())}")
                context = input("Context (default): ").strip() or "default"
            else:
                context = "default"
            
            prompt = input("What should I write about? ")
            
            if context in contexts:
                result = engine.generate_with_context(prompt, context)
            else:
                result = engine.generate_with_tone(prompt)
            
            print(f"\n📝 Generated ({context}):\n{result}")
        
        elif command == "evolve":
            try:
                days = int(input("Days to plot (30): ").strip() or "30")
                context = input("Context (default): ").strip() or "default"
                engine.plot_voice_evolution(days, context)
            except ValueError:
                print("Invalid number of days")
        
        elif command == "trends":
            try:
                days = int(input("Days to analyze (7): ").strip() or "7")
                trends = engine.analyze_tone_trends(days)
                
                if "error" in trends:
                    print(trends["error"])
                else:
                    print(f"\n📊 Tone Trends ({trends['date_range']}):")
                    print(f"Days analyzed: {trends['total_days']}")
                    print(f"Contexts: {', '.join(trends['contexts_analyzed'])}")
                    print(f"Most active day: {trends['most_active_day']}")
                    print(f"Top tones: {trends['dominant_tones']}")
                    
                    avg_intensity = np.mean([t['intensity'] for t in trends['intensity_trend']])
                    print(f"Average intensity: {avg_intensity:.1f}/10")
            except ValueError:
                print("Invalid number of days")
        
        elif command == "export":
            filename = input("Export filename (auto-generated): ").strip()
            path = engine.export_profile(filename if filename else None)
            print(f"✅ Profile exported to: {path}")
        
        elif command == "import":
            filepath = input("Import filepath: ").strip()
            try:
                engine.import_profile(filepath)
            except Exception as e:
                print(f"Error importing: {e}")
        
        elif command == "recommend":
            recommendations = engine.get_voice_recommendations()
            print("\n💡 Voice Model Recommendations:")
            for rec in recommendations:
                print(f"  {rec}")
        
        else:
            print("Unknown command. Try: process, batch, context, generate, evolve, trends, export, import, recommend, quit")

if __name__ == "__main__":
    advanced_cli() 