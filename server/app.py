import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env from project root
load_dotenv()

# Import ToneEngine from the project
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "tone_engine"))
from main import ToneEngine  # type: ignore

# Optional DB helper to list clients if available
try:
    from tone_engine.db import DB  # when imported as package
except Exception:
    try:
        from db import DB  # when running from tone_engine path
    except Exception:
        DB = None  # type: ignore

app = FastAPI(title="Tone Engine API", version="0.1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RephraseRequest(BaseModel):
    client: str = "default"
    text: str


class EnsureClientRequest(BaseModel):
    email: str


class GenerateBriefRequest(BaseModel):
    client: str = "default"
    content_type: str = "post"  # blog|email|post|thread
    audience: str = ""
    objective: str = ""
    length: str = "medium"  # short|medium|long
    must: list[str] = []
    avoid: list[str] = []
    prompt: str = ""
    platform: str = "generic"  # generic|twitter|linkedin|x


def normalize_client(name: str) -> str:
    return name.strip().lower()


def get_engine(client: str) -> ToneEngine:
    return ToneEngine(base_dir="tone_engine", client_name=client)


def build_generation_prompt(profile: dict, req: GenerateBriefRequest) -> str:
    length_map_words = {
        "short": "~120-200 words",
        "medium": "~400-700 words",
        "long": "~900-1300 words",
    }
    exemplars = "\n".join(
        [ex.get("sample_text", "") for ex in profile.get("recent_examples", [])[-3:]]
    )

    # Platform-aware formatting guidance
    platform = (req.platform or "generic").lower()
    if platform == "x":
        platform = "twitter"

    if platform == "twitter":
        platform_rules = """
Platform: Twitter/X
Format rules:
- Write as a single tweet or concise mini-thread (1–3 tweets max).
- Hook in the first sentence. Strong claim or contrast. Avoid fluff.
- Keep sentences punchy; prefer line breaks between thoughts.
- No hashtags in the body; at most 1–2 at the end if truly helpful.
- Avoid emojis unless they add clarity. No clickbait.
- Keep under ~240 characters for a single tweet; for 2–3 tweets, ~240 each.
"""
    elif platform == "linkedin":
        platform_rules = """
Platform: LinkedIn
Format rules:
- Hook first line to stop the scroll; then 2–5 short paragraphs.
- Use whitespace for readability; 1–3 sentences per paragraph.
- No hashtags at the top; at most 2–3 at the end if relevant.
- Keep jargon light; be specific and practical. Professional but conversational.
- Emojis sparingly, only to improve scannability.
"""
    else:
        platform_rules = """
Platform: Generic
Format rules:
- Clear structure with a strong opening, crisp body, and specific call to action.
- Keep to the requested length and avoid filler.
"""

    length_target = length_map_words.get(req.length, "medium")

    return f"""
You are a professional ghostwriter. Match the client's voice exactly: vocabulary, cadence, rhythm, and rhetorical habits. Do not add facts not present in the brief. Respect constraints strictly.

VOICE PROFILE
Primary Tone: {profile.get('dominant_traits', {}).get('primary_tone', 'conversational')}
Top Style Patterns: {', '.join(profile.get('dominant_traits', {}).get('top_style_patterns', []))}
Recent Exemplars:
{exemplars}

BRIEF
Content Type: {req.content_type}
Audience: {req.audience}
Objective: {req.objective}
Length Target: {length_target}
Must include: {', '.join(req.must)}
Avoid: {', '.join(req.avoid)}

{platform_rules}

TASK
{req.prompt}

Write in the exact client voice and style. Obey the platform format rules above and keep within the length target or platform limits.
"""


@app.get("/")
def landing():
    landing_path = Path(__file__).resolve().parent / "static" / "landing.html"
    if landing_path.exists():
        return FileResponse(str(landing_path))
    # Fallback to app if landing not present
    return RedirectResponse(url="/app/")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ensureClient")
def ensure_client(req: EnsureClientRequest):
    client_name = normalize_client(req.email)
    # Prefer DB upsert
    try:
        if 'DB' in globals() and DB is not None:
            db = DB()  # type: ignore
            if getattr(db, 'enabled')():
                cid = db.upsert_client(client_name)  # type: ignore
                return {"client": client_name, "created": bool(cid)}
    except Exception:
        pass
    # Ensure local dirs so the client exists locally
    engine = get_engine(client_name)
    engine.transcripts_dir.mkdir(parents=True, exist_ok=True)
    engine.profiles_dir.mkdir(parents=True, exist_ok=True)
    return {"client": client_name, "created": True}


@app.get("/api/clients")
def list_clients(email: Optional[str] = None):
    if email:
        name = normalize_client(email)
        # Prefer DB check
        try:
            if 'DB' in globals() and DB is not None:
                db = DB()  # type: ignore
                if getattr(db, 'enabled')():
                    res = db.client.table("clients").select("name").eq("name", name).execute()
                    if res.data:
                        return {"clients": [name]}
                    return {"clients": []}
        except Exception:
            pass
        # Fallback local check
        prof = ROOT_DIR / "tone_engine" / "clients" / name / "profiles" / "tone_profile.json"
        return {"clients": [name] if prof.exists() else [name]}

    # No email provided: list all known (DB or default)
    try:
        if 'DB' in globals() and DB is not None:
            db = DB()  # type: ignore
            if getattr(db, 'enabled')():
                res = db.client.table("clients").select("name").order("name").execute()
                names = [r["name"] for r in (res.data or [])]
                return {"clients": names}
    except Exception:
        pass

    base_clients_dir = ROOT_DIR / "tone_engine" / "clients"
    names = [p.name for p in base_clients_dir.iterdir()] if base_clients_dir.exists() else []
    return {"clients": names or ["default"]}


@app.get("/api/profile")
def get_profile(client: str = "default"):
    engine = get_engine(client)
    profile_path = engine.tone_profile_path
    if not profile_path.exists():
        raise HTTPException(status_code=404, detail="No tone profile found for this client")
    import json
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.post("/api/rephrase")
def rephrase(req: RephraseRequest):
    engine = get_engine(req.client)
    result = engine.rephrase_in_my_tone(req.text)
    return {"result": result}


@app.post("/api/processTranscript")
def process_transcript(client: str = Form("default"), file: UploadFile = File(...)):
    engine = get_engine(client)
    transcripts_dir = engine.transcripts_dir
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    dest_path = transcripts_dir / file.filename
    with dest_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        chunks = engine.process_transcript(file.filename)
        profile = engine.update_tone_profile(chunks)
        return {
            "chunks": len(chunks),
            "profile_total_chunks": profile.get("total_chunks", 0),
            "message": "Processed and profile updated"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/generateWithBrief")
def generate_with_brief(req: GenerateBriefRequest):
    engine = get_engine(req.client)
    # Load profile (reusing generate_with_tone fallback logic)
    # Build a temporary prompt and route through Anthropic directly to keep low temperature.
    # Reuse engine.anthropic_client
    if not engine.anthropic_client:
        raise HTTPException(status_code=400, detail="Anthropic key missing. Set ANTHROPIC_API_KEY in .env")

    # Load profile via engine.generate_with_tone's internal logic by emulating the file/DB fetch
    # Refactor: read profile file or DB directly
    profile = None
    if engine.tone_profile_path.exists():
        import json
        with open(engine.tone_profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    elif engine.client_id and engine.db and getattr(engine.db, 'enabled')():
        try:
            res = engine.db.client.table("tone_profiles").select("profile").eq("client_id", engine.client_id).single().execute()
            if res.data and res.data.get("profile"):
                profile = res.data["profile"]
        except Exception:
            pass
    if not profile:
        raise HTTPException(status_code=404, detail="No tone profile found for this client")

    prompt = build_generation_prompt(profile, req)
    try:
        resp = engine.anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,
            temperature=0.25,
            system="You are a careful ghostwriter matching the client’s exact voice and constraints.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        # Log to DB if available
        try:
            if engine.client_id and engine.db and getattr(engine.db, 'enabled')():
                engine.db.insert_generation(engine.client_id, {
                    "content_type": req.content_type,
                    "audience": req.audience,
                    "objective": req.objective,
                    "length": req.length,
                    "must": req.must,
                    "avoid": req.avoid,
                    "prompt": req.prompt,
                    "output": text,
                })
        except Exception:
            pass
        return {"result": text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Serve static frontend under /app
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")