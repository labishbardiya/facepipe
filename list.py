#!/usr/bin/env python3
"""
list.py — List all currently enrolled persons in the facial recognition database.

Usage:
    python list.py

Prints each enrolled name and how many raw embeddings are stored for them.
More embeddings generally means a more robust averaged representation.
"""

from db_manager import get_enrollment_summary


def main() -> None:
    summary = get_enrollment_summary()

    if not summary:
        print("[!] No enrolled faces found.")
        print("    Run: python enroll.py \"Person Name\"  to enroll someone.")
        return

    print(f"[✓] {len(summary)} enrolled person(s):\n")
    print(f"  {'Name':<30} {'Embeddings':>10}")
    print(f"  {'─' * 30} {'─' * 10}")

    for name, count in sorted(summary.items()):
        print(f"  {name:<30} {count:>10}")

    print()


if __name__ == "__main__":
    main()
