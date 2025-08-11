import os
import json
import streamlit as st
from dotenv import load_dotenv

# Supabase client
try:
    from supabase import create_client
except Exception as e:  # pragma: no cover
    create_client = None  # type: ignore

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or ""
)

st.set_page_config(page_title="Tone Engine Dashboard", layout="wide")

st.title("Tone Engine Dashboard")

if not SUPABASE_URL or not SUPABASE_KEY or create_client is None:
    st.error("Supabase not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY (or SERVICE_ROLE_KEY) in .env")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Auth state
if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.header("Sign in")
    if st.session_state.user is None:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state.user = res.user
                st.rerun()
            except Exception as e:  # pragma: no cover
                st.error(f"Sign-in failed: {e}")
    else:
        st.success(f"Signed in as {st.session_state.user.email}")
        if st.button("Sign out"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.user = None
            st.rerun()

if st.session_state.user is None:
    st.info("Please sign in to view your profiles and transcripts.")
    st.stop()

# Fetch clients
clients_res = supabase.table("clients").select("id,name").order("name").execute()
clients = clients_res.data or []

if not clients:
    st.warning("No clients found. Process transcripts locally first to upsert a client, or insert via DB.")
    st.stop()

client_names = {c["name"]: c["id"] for c in clients}
selected_name = st.selectbox("Client", list(client_names.keys()))
client_id = client_names[selected_name]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Tone Profile")
    prof_res = supabase.table("tone_profiles").select("profile,updated_at").eq("client_id", client_id).single().execute()
    if prof_res.data:
        profile = prof_res.data.get("profile", {})
        st.caption(f"Updated at: {prof_res.data.get('updated_at')}")
        st.json(profile)
    else:
        st.info("No tone profile found yet for this client.")

with col2:
    st.subheader("Recent Chunks")
    # Get last 10 chunks via transcripts join
    # First fetch transcript ids
    ts_res = supabase.table("transcripts").select("id,filename,created_at").eq("client_id", client_id).order("created_at", desc=True).limit(5).execute()
    transcripts = ts_res.data or []
    if not transcripts:
        st.info("No transcripts for this client.")
    else:
        for t in transcripts:
            st.markdown(f"**{t['filename']}** · {t['created_at']}")
            ch_res = supabase.table("chunks").select("chunk_id,ts,tone:analysis->>tone,chunk_text").eq("transcript_id", t["id"]).order("ts", desc=True).limit(5).execute()
            for ch in ch_res.data or []:
                with st.expander(f"{ch['chunk_id']} · {ch.get('tone')}"):
                    st.write(ch.get("ts"))
                    # show a short preview
                    preview = (ch.get("chunk_text") or "")[:300]
                    st.write(preview + ("..." if len(preview) == 300 else ""))