#!/usr/bin/env python3
"""MLE Hiring Challenge — Support Triage Agent.

Usage:
    python code/main.py                    # Process all tickets
    python code/main.py --dry-run          # Skip LLM calls (for testing)
    python code/main.py --ticket-index 5   # Process single ticket (0-based)
    python code/main.py --concurrency 10   # Override concurrency limit
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Add code/ to path
sys.path.insert(0, os.path.dirname(__file__))

from io_csv import read_tickets, write_output
from agent import process_ticket
from llm import set_max_concurrency


async def run_all(
    tickets: list[dict],
    dry_run: bool = False,
    concurrency: int = 20,
) -> list[dict]:
    """Process all tickets concurrently with a semaphore limit."""
    sem = asyncio.Semaphore(concurrency)

    async def bounded(row: dict) -> dict:
        async with sem:
            return await process_ticket(row, dry_run=dry_run)

    tasks = [bounded(row) for row in tickets]
    results = await asyncio.gather(*tasks)
    return list(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls; use canned responses (fast, for testing structure)"
    )
    parser.add_argument(
        "--ticket-index", type=int, default=None,
        help="Process only ticket at this 0-based index"
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent in-flight API calls (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Override output CSV path"
    )
    args = parser.parse_args()

    print("Loading tickets...")
    tickets = read_tickets()
    total = len(tickets)
    print(f"  {total} tickets loaded.")

    if args.ticket_index is not None:
        if args.ticket_index >= total:
            print(f"ERROR: ticket-index {args.ticket_index} >= total {total}", file=sys.stderr)
            sys.exit(1)
        tickets = [tickets[args.ticket_index]]
        print(f"  Processing single ticket at index {args.ticket_index}")

    set_max_concurrency(args.concurrency)

    start = time.time()
    print(f"Processing {'(dry-run)' if args.dry_run else ''}...")

    results = asyncio.run(run_all(tickets, dry_run=args.dry_run, concurrency=args.concurrency))

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed / len(tickets):.2f}s per ticket)")

    if args.output:
        # Override output path (for single-ticket debug)
        import csv
        from io_csv import OUTPUT_COLUMNS
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, quoting=2, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                at = row.get("actions_taken", "[]")
                if isinstance(at, list):
                    row["actions_taken"] = json.dumps(at)
                writer.writerow(row)
        print(f"  Output written to {args.output}")
    else:
        write_output(results)
        from io_csv import OUTPUT_CSV
        print(f"  Output written to {OUTPUT_CSV}")

    # Print summary
    statuses = [r.get("status", "?") for r in results]
    replied = statuses.count("replied")
    escalated = statuses.count("escalated")
    print(f"\nSummary: {replied} replied, {escalated} escalated, {len(results)} total")

    if elapsed > 160:
        print(f"WARNING: {elapsed:.0f}s is close to the 3-minute (180s) limit!", file=sys.stderr)


if __name__ == "__main__":
    main()
