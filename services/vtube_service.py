from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from core.logger import get_logger
from core.state import AppState
from services.base_service import BaseService

try:
    import websocket
except ImportError:
    websocket = None


class VTubeService(BaseService):
    name = "vtube"

    def __init__(
        self,
        runtime,
        url: str = "ws://127.0.0.1:8001",
        plugin_name: str = "Neuro AI Assistant",
        developer: str = "Local User",
        token_file: Optional[str] = None,
    ):
        super().__init__(runtime)
        self.state: AppState = runtime.state
        self.logger = get_logger("services.vtube")
        self.url = url
        self.plugin_name = plugin_name
        self.developer = developer
        self.token_file = Path(token_file) if token_file else None
        self.ws = None
        self.hotkeys = []
        self.request_number = 0

    def start(self) -> None:
        if websocket is None:
            self.logger.warning("websocket-client not installed; VTube support disabled")
            self.state.set_service_status(self.name, "disabled")
            return

        try:
            self.connect()
            self.state.set_service_status(self.name, "running")
        except Exception as exc:
            self.logger.exception("VTube connection failed: %s", exc)
            self.state.set_service_status(self.name, "failed")

    def stop(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = None
        self.state.set_service_status(self.name, "stopped")

    def _request_id(self) -> str:
        self.request_number += 1
        return f"NeuroAI{self.request_number}"

    def _send_request(self, message_type: str, data: Optional[dict] = None, timeout: int = 8):
        if self.ws is None:
            raise ConnectionError("VTube Studio API is not connected")

        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": self._request_id(),
            "messageType": message_type,
        }
        if data is not None:
            payload["data"] = data

        self.ws.settimeout(timeout)
        self.ws.send(json.dumps(payload, ensure_ascii=False))

        while True:
            response = json.loads(self.ws.recv())
            if response.get("requestID") == payload["requestID"]:
                if response.get("messageType") == "APIError":
                    details = response.get("data", {}).get("message", "API error")
                    raise RuntimeError(details)
                return response

    def connect(self) -> bool:
        if websocket is None:
            return False

        self.ws = websocket.create_connection(self.url, timeout=5)
        token = ""
        if self.token_file is not None and self.token_file.exists():
            token = self.token_file.read_text(encoding="utf-8").strip()

        if not token:
            response = self._send_request(
                "AuthenticationTokenRequest",
                {
                    "pluginName": self.plugin_name,
                    "pluginDeveloper": self.developer,
                },
                timeout=90,
            )
            token = response.get("data", {}).get("authenticationToken", "")
            if not token:
                raise RuntimeError("Could not obtain VTube Studio authentication token")

            if self.token_file is not None:
                self.token_file.parent.mkdir(parents=True, exist_ok=True)
                self.token_file.write_text(token, encoding="utf-8")

        auth = self._send_request(
            "AuthenticationRequest",
            {
                "pluginName": self.plugin_name,
                "pluginDeveloper": self.developer,
                "authenticationToken": token,
            },
            timeout=10,
        )
        if not auth.get("data", {}).get("authenticated", False):
            raise RuntimeError("VTube Studio authentication failed")

        hotkey_response = self._send_request("HotkeysInCurrentModelRequest", timeout=8)
        self.hotkeys = hotkey_response.get("data", {}).get("availableHotkeys", [])

        self.logger.info("Connected to VTube Studio")
        return True

    def trigger_hotkey(self, hotkey_name: str) -> None:
        if not hotkey_name:
            return

        try:
            if self.ws is None:
                if not self.connect():
                    return

            if not self.hotkeys:
                response = self._send_request("HotkeysInCurrentModelRequest", timeout=8)
                self.hotkeys = response.get("data", {}).get("availableHotkeys", [])

            match = next(
                (h for h in self.hotkeys if h.get("name", "").casefold() == hotkey_name.casefold()),
                None,
            )
            if match is None:
                self.logger.warning("Hotkey not found: %s", hotkey_name)
                return

            self._send_request("HotkeyTriggerRequest", {"hotkeyID": match["hotkeyID"]}, timeout=5)
        except Exception as exc:
            self.logger.exception("Hotkey trigger failed: %s", exc)
            self.ws = None
            self.hotkeys = []