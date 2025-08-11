import os
from dotenv import load_dotenv
from supabase import create_client


def main() -> None:
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    print("URL set:", bool(url), "KEY set:", bool(key))
    if not url or not key:
        print("Missing SUPABASE_URL or key in .env")
        return

    client = create_client(url, key)
    try:
        res = client.table("clients").select("id").limit(1).execute()
        count = len(res.data) if res.data else 0
        print("Connected OK. clients rows:", count)
    except Exception as e:
        print("Connected but query failed (likely missing tables or RLS):", str(e)[:300])


if __name__ == "__main__":
    main()