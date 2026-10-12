"""Print a join token for a test call from the browser.

    python -m agent.make_token --lead L003

Open https://agents-playground.livekit.io (Chrome) -> "Manual" -> paste the URL and token.
The room name "lead-L003" tells the agent which CRM lead it is calling.
"""

from __future__ import annotations

import argparse
import os
import time

from dotenv import load_dotenv
from livekit import api


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--lead", default=os.getenv("DEFAULT_LEAD_ID", "L001"))
    p.add_argument("--identity", default="tester")
    args = p.parse_args()

    room = f"lead-{args.lead}-{int(time.time())}"
    token = (
        api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(args.identity)
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    print(f"LiveKit URL : {os.getenv('LIVEKIT_URL', 'ws://localhost:7880')}")
    print(f"Room        : {room}")
    print(f"Token       : {token}")


if __name__ == "__main__":
    main()
