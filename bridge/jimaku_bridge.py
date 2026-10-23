from __future__ import annotations
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import json
import re

class JimakuBridgeHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Placeholder: serve /v2/index.html and /health
        pass

    def do_POST(self):
        if self.path == "/api/recognized":
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length)
            data = json.loads(payload.decode("utf-8"))
            text = data.get("text", "").strip()
            if text:
                # emit event to runtime or event bus
                pass
            self.send_response(200)
            self.end_headers()

def start_jimaku_bridge(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), JimakuBridgeHandler)
    return server