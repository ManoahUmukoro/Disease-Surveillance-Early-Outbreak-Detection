"""
stream.py — the INGESTION BUS (Redis Streams) of the Adaptive SVM surveillance system.

In the Chapter 4 architecture this is the lightweight substitute for Kafka: when a case is
registered, an event is published to the `surveillance:cases` stream so downstream processing
(the adaptive learner, aggregators) can consume it "as data arrives".

Graceful degradation: if redis isn't installed or no REDIS_URL is configured, `available` is
False and publish_case() returns the exact event payload it WOULD have sent — so the UI can still
show the streaming step transparently. Under Docker Compose a real Redis service runs.
"""
import json
import os
from datetime import datetime

STREAM = "surveillance:cases"


def _url():
    url = os.environ.get("REDIS_URL")
    if not url:
        try:
            import streamlit as st
            url = st.secrets.get("REDIS_URL", None)
        except Exception:
            url = None
    return url


class Stream:
    """Never raises on connection failure — check `.available`."""

    def __init__(self):
        self.available = False
        self.reason = ""
        self._r = None
        url = _url()
        if not url:
            self.reason = "no REDIS_URL configured (event shown, not published)"
            return
        try:
            import redis
            self._r = redis.from_url(url, socket_connect_timeout=3)
            self._r.ping()
            self.available = True
        except Exception as e:
            self.reason = f"{type(e).__name__}: {e}"

    def publish_case(self, case_id, payload: dict):
        """Publish a new-case event. Returns a result dict that always includes the `event`
        payload, whether or not a real Redis was reachable."""
        event = {"case_id": case_id, "ts": datetime.utcnow().isoformat(), **payload}
        if self.available:
            flat = {k: (json.dumps(v) if isinstance(v, (list, dict)) else str(v))
                    for k, v in event.items()}
            msg_id = self._r.xadd(STREAM, flat)
            return {"published": True, "stream": STREAM,
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                    "event": event}
        return {"published": False, "reason": self.reason, "stream": STREAM, "event": event}

    def length(self):
        if not self.available:
            return None
        try:
            return self._r.xlen(STREAM)
        except Exception:
            return None
