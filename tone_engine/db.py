import os
from typing import Optional

try:
    from supabase import create_client, Client
except Exception as import_error:  # pragma: no cover
    create_client = None  # type: ignore
    Client = None  # type: ignore


class DB:
    """Thin Supabase helper. If creds are missing or client import fails, behaves as disabled."""

    def __init__(self):
        self._client: Optional[Client] = None
        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
        )
        if url and key and create_client is not None:
            try:
                self._client = create_client(url, key)
            except Exception as e:  # pragma: no cover
                print(f"⚠️  Supabase client init failed: {e}")
                self._client = None

    def enabled(self) -> bool:
        return self._client is not None

    @property
    def client(self) -> Client:
        if not self._client:
            raise RuntimeError("Supabase client not initialized")
        return self._client

    def upsert_client(self, name: str) -> Optional[str]:
        if not self.enabled():
            return None
        # Perform upsert, then fetch id via a separate select (supabase-py upsert does not support chaining .select)
        self.client.table("clients").upsert({"name": name}, on_conflict="name").execute()
        res = (
            self.client
            .table("clients")
            .select("id")
            .eq("name", name)
            .single()
            .execute()
        )
        return res.data["id"] if res.data else None

    def ensure_transcript(self, client_id: str, filename: str) -> Optional[str]:
        if not self.enabled():
            return None
        self.client.table("transcripts").upsert(
            {"client_id": client_id, "filename": filename},
            on_conflict="client_id,filename",
        ).execute()
        res = (
            self.client
            .table("transcripts")
            .select("id")
            .eq("client_id", client_id)
            .eq("filename", filename)
            .single()
            .execute()
        )
        return res.data["id"] if res.data else None

    def upsert_chunk(self, transcript_id: str, chunk: dict) -> None:
        if not self.enabled():
            return
        self.client.table("chunks").upsert(
            {
                "transcript_id": transcript_id,
                "chunk_id": chunk["chunk_id"],
                "source_file": chunk.get("source_file"),
                "ts": chunk["timestamp"],
                "chunk_text": chunk.get("chunk_text", ""),
                "analysis": chunk.get("analysis", {}),
            },
            on_conflict="chunk_id",
        ).execute()

    def upsert_tone_profile(self, client_id: str, profile: dict) -> None:
        if not self.enabled():
            return
        self.client.table("tone_profiles").upsert(
            {"client_id": client_id, "profile": profile},
            on_conflict="client_id",
        ).execute()

    def insert_generation(self, client_id: str, payload: dict) -> None:
        if not self.enabled():
            return
        try:
            self.client.table("generations").insert({
                "client_id": client_id,
                **payload
            }).execute()
        except Exception as e:
            # Table may not exist; ignore in MVP
            print(f"⚠️  DB generation log failed: {str(e)[:100]}")