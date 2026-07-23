#!/usr/bin/env python3
"""
stream_consumer.py — the streaming processing loop (Redis Streams consumer).

This is the "process the incoming stream continuously / behave as data arrives" component of the
Chapter 4 architecture, running as its own process (in Docker Compose it is the `consumer` service).
It blocks on the `surveillance:cases` stream and reacts to each new case the dashboard registers.

Here it logs and tallies each event — but this is exactly the seam where an online `partial_fit()`
update of the Adaptive SVM, or a live aggregation into surveillance_events, would run.

Run standalone:  REDIS_URL=redis://localhost:6379 python scripts/stream_consumer.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
STREAM = "surveillance:cases"


def main():
    url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        import redis
    except ImportError:
        print("redis not installed — run `pip install redis`")
        return
    r = redis.from_url(url)
    try:
        r.ping()
    except Exception as e:
        print(f"[consumer] cannot reach Redis at {url}: {e}")
        return

    print(f"[consumer] listening on '{STREAM}' at {url} …")
    last, seen = "$", 0
    while True:
        resp = r.xread({STREAM: last}, block=5000, count=10)
        if not resp:
            continue
        for _stream, messages in resp:
            for msg_id, fields in messages:
                last = msg_id
                seen += 1
                d = {k.decode(): v.decode() for k, v in fields.items()}
                print(f"[consumer] #{seen} case={d.get('case_id')} "
                      f"{d.get('state')}/{d.get('disease')} risk={d.get('risk_level')} "
                      f"→ (partial_fit / aggregation would run here)")


if __name__ == "__main__":
    main()
