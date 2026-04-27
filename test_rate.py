#!/usr/bin/env python3
"""
test_openai_limits.py
Fires N requests at the OpenAI API and reports how many succeed vs. fail.
"""

import time
import os
from typing import Optional
from dataclasses import dataclass, field
import dotenv

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    raise SystemExit("openai package not found. Run: pip install openai")

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL   = "gpt-4o-mini"
TOTAL_REQUESTS = 20          # total requests to attempt
DELAY_BETWEEN  = 0.0         # seconds between requests (0 = fire as fast as possible)
TEST_PROMPT    = "Say 'ok' in one word."

dotenv.load_dotenv()

# ── Result tracking ───────────────────────────────────────────────────────────

@dataclass
class Results:
    successes: int = 0
    rate_limits: int = 0
    other_errors: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

# ── Helpers ───────────────────────────────────────────────────────────────────

def test_generate(client: OpenAI, results: Results, idx: int) -> None:
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=GEN_MODEL,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=5,
        )
        elapsed = time.perf_counter() - t0
        results.successes += 1
        results.latencies.append(elapsed)
        print(f"  [{idx:>3}] ✓  {elapsed:.2f}s  → {response.choices[0].message.content!r}")
    except RateLimitError as e:
        results.rate_limits += 1
        msg = str(e)[:80]
        results.errors.append(f"[{idx}] RateLimitError: {msg}")
        print(f"  [{idx:>3}] ✗  RATE LIMIT — {msg}")
    except APIError as e:
        results.other_errors += 1
        msg = str(e)[:80]
        results.errors.append(f"[{idx}] APIError: {msg}")
        print(f"  [{idx:>3}] ✗  API ERROR  — {msg}")

def test_embed(client: OpenAI, results: Results, idx: int) -> None:
    t0 = time.perf_counter()
    try:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input="test embedding request",
        )
        elapsed = time.perf_counter() - t0
        results.successes += 1
        results.latencies.append(elapsed)
        dims = len(response.data[0].embedding)
        print(f"  [{idx:>3}] ✓  {elapsed:.2f}s  → embedding dim={dims}")
    except RateLimitError as e:
        results.rate_limits += 1
        msg = str(e)[:80]
        results.errors.append(f"[{idx}] RateLimitError: {msg}")
        print(f"  [{idx:>3}] ✗  RATE LIMIT — {msg}")
    except APIError as e:
        results.other_errors += 1
        msg = str(e)[:80]
        results.errors.append(f"[{idx}] APIError: {msg}")
        print(f"  [{idx:>3}] ✗  API ERROR  — {msg}")

def print_summary(label: str, results: Results, total: int, wall: float) -> None:
    print(f"\n{'─'*55}")
    print(f"  {label} — summary ({total} attempts, {wall:.1f}s wall time)")
    print(f"{'─'*55}")
    print(f"  Successes  : {results.successes}")
    print(f"  Rate limits: {results.rate_limits}")
    print(f"  Other errors: {results.other_errors}")
    if results.latencies:
        avg = sum(results.latencies) / len(results.latencies)
        print(f"  Avg latency: {avg:.2f}s  (successful requests only)")
    if results.errors:
        print(f"\n  Error log:")
        for e in results.errors:
            print(f"    {e}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in environment.")

    client = OpenAI(api_key=api_key)

    # ── Generation test ───────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Generation test — {TOTAL_REQUESTS} requests → {GEN_MODEL}")
    print(f"{'═'*55}")
    gen_results = Results()
    t_start = time.perf_counter()
    for i in range(1, TOTAL_REQUESTS + 1):
        test_generate(client, gen_results, i)
        if DELAY_BETWEEN > 0:
            time.sleep(DELAY_BETWEEN)
    gen_wall = time.perf_counter() - t_start
    print_summary("Generation", gen_results, TOTAL_REQUESTS, gen_wall)

    # ── Embedding test ────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  Embedding test — {TOTAL_REQUESTS} requests → {EMBED_MODEL}")
    print(f"{'═'*55}")
    emb_results = Results()
    t_start = time.perf_counter()
    for i in range(1, TOTAL_REQUESTS + 1):
        test_embed(client, emb_results, i)
        if DELAY_BETWEEN > 0:
            time.sleep(DELAY_BETWEEN)
    emb_wall = time.perf_counter() - t_start
    print_summary("Embedding", emb_results, TOTAL_REQUESTS, emb_wall)

    # ── Overall ───────────────────────────────────────────────────────────────
    total_ok   = gen_results.successes + emb_results.successes
    total_fail = gen_results.rate_limits + gen_results.other_errors + \
                 emb_results.rate_limits + emb_results.other_errors
    print(f"\n{'═'*55}")
    print(f"  OVERALL: {total_ok} succeeded, {total_fail} failed out of {TOTAL_REQUESTS*2}")
    print(f"{'═'*55}\n")

if __name__ == "__main__":
    main()