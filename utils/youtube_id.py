from __future__ import annotations
from urllib.parse import urlparse, parse_qs

def extract_youtube_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip().rstrip(".,。!?！？;；)）]」』"))
        host = (parsed.hostname or "").lower()

        if host == "youtu.be":
            return parsed.path.strip("/").split("/")[0]

        if host in ("youtube.com", "m.youtube.com") or host.endswith(".youtube.com"):
            if parsed.path == "/watch":
                return parse_qs(parsed.query).get("v", [""])[0]

            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live"):
                return parts[1]
    except Exception:
        pass
    return None