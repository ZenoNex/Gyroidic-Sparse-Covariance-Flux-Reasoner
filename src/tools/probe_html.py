"""
Full smoke test: index chat.html and verify real titles appear + messages parse.
"""
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import mmap
from src.tools.fast_chat_viewer import ConversationIndex, MessageParser

DATA_DIR = (
    r"data\raw\chatgpt_userIla_and_archetpes_dyads_data"
    r"\9894d8be355693bad4f30a9a8341f63f0519577efadeafd6e93ad9c97521d980-2026-03-31-10-43-19-a178149902ef4042a44540feb4301932"
)
html_path = os.path.join(DATA_DIR, "chat.html")

print(f"Opening: {html_path}")

t0 = time.perf_counter()
with open(html_path, "rb") as fh:
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    file_size = len(mm)
    print(f"File size: {file_size / 1024 / 1024:.1f} MB")

    idx = ConversationIndex(mm, file_size)
    idx.build()
    t1 = time.perf_counter()
    print(f"Indexed {len(idx.entries)} conversations in {t1-t0:.2f}s")

    named = sum(1 for t, _, _ in idx.entries if not t.startswith("Conversation "))
    print(f"Conversations with real titles: {named}/{len(idx.entries)}")

    print("\nFirst 10 conversation titles:")
    for i, (title, start, end) in enumerate(idx.entries[:10]):
        size_kb = (end - start) / 1024
        print(f"  [{i:4d}] {size_kb:7.1f} KB  {title!r}")

    print("\nParsing first 5 conversations...")
    for i in range(min(5, len(idx.entries))):
        title, start, end = idx.entries[i]
        mm.seek(start)
        raw = mm.read(end - start)
        t2 = time.perf_counter()
        messages = MessageParser.parse(raw)
        t3 = time.perf_counter()
        print(f"\n  [{i}] {title!r}  ({len(messages)} msgs, parsed in {(t3-t2)*1000:.1f}ms)")
        for role, text in messages[:2]:
            print(f"    [{role:9}] {text[:90]!r}")

    mm.close()

print("\nSmoke test PASSED.")
