from __future__ import annotations

import json
import re
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from services.base_service import BaseService

_JIMAKU_RECOGNITION_BRIDGE = r"""
<script id="neuro-ai-recognition-bridge">
(() => {
  if (window.__neuroAiRecognitionBridgeInstalled) return;
  window.__neuroAiRecognitionBridgeInstalled = true;

  const sendFinal = (event) => {
    try {
      if (!event || !event.results) return;
      const parts = [];
      for (let i = 0; i < event.results.length; i++) {
        const result = event.results[i];
        if (result && result.isFinal && result[0] && result[0].transcript) {
          parts.push(String(result[0].transcript).trim());
        }
      }
      const text = parts.join('').trim();
      if (!text) return;
      fetch('/api/recognized', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        keepalive: true
      }).catch((err) => console.warn('[NeuroAI bridge] send failed:', err));
    } catch (err) {
      console.warn('[NeuroAI bridge] result processing error:', err);
    }
  };

  const patch = (Ctor) => {
    if (!Ctor || !Ctor.prototype || typeof Ctor.prototype.start !== 'function') return false;
    const proto = Ctor.prototype;
    if (proto.__neuroAiStartPatched) return true;
    const originalStart = proto.start;
    try {
      Object.defineProperty(proto, '__neuroAiStartPatched', { value: true, configurable: true });
      proto.start = function (...args) {
        if (!this.__neuroAiResultListenerAdded) {
          try {
            EventTarget.prototype.addEventListener.call(this, 'result', sendFinal);
            Object.defineProperty(this, '__neuroAiResultListenerAdded', { value: true, configurable: true });
          } catch (error) {
            console.warn('[NeuroAI bridge] add result handler failed:', error);
          }
        }
        return originalStart.apply(this, args);
      };
      return true;
    } catch (error) {
      console.warn('[NeuroAI bridge] patch start failed:', error);
      return false;
    }
  };

  let attempts = 0;
  const retry = () => {
    attempts++;
    const standard = patch(window.SpeechRecognition);
    const webkit = patch(window.webkitSpeechRecognition);
    if ((standard || webkit) || attempts >= 40) return;
    window.setTimeout(retry, 250);
  };
  retry();
})();
</script>
"""

class JimakuBridgeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self._runtime = kwargs.pop("runtime", None)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        route = self.path.split("?", 1)[0]

        if route == "/health":
            payload = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/recognized":
            self.send_error(404, "Unknown endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = str(payload.get("text", "")).strip()
            if not text:
                self.send_error(400, "Empty text")
                return

            if self._runtime is not None:
                # insert event published to runtime
                self._runtime.emit("recognized_text", text)

            response = json.dumps({"ok": True}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except Exception as exc:
            self.send_error(400, f"Invalid JSON: {exc}")

class JimakuBridgeService(BaseService):
    name = "jimaku_bridge"

    def __init__(self, runtime, port: int = 8765, root_dir: str = "."):
        super().__init__(runtime)
        self.logger = get_logger("services.jimaku_bridge")
        self.port = port
        self.root_dir = root_dir
        self.server = None

    def start(self) -> None:
        def handler(*args, **kwargs):
            return JimakuBridgeHandler(*args, runtime=self.runtime, **kwargs)

        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.server.serve_forever()
        self.state.set_service_status(self.name, "running")

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        self.state.set_service_status(self.name, "stopped")