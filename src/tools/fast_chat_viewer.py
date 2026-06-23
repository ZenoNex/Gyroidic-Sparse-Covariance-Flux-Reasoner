"""
Fast Chat Viewer.
A Tkinter-based lazy-loading viewer for massive ChatGPT data exports.

Strategy: The chat.html is a 342 MB <script> tag containing a JSON array
of ~5783 conversation objects. The <body> is an empty <div id='root'>.
We index the file using mmap to find each conversation object boundary
(via the '{"async_status":' sentinel), then on-demand parse each individual
conversation from JSON and render it message-by-message in batches.

This avoids:
  - Loading the full file into RAM
  - Inserting large text blobs into Tkinter (which freezes it)
  - Blocking the main thread during indexing or parsing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mmap
import re
import os
import json
import threading
from typing import List, Tuple, Optional

# Sentinel byte-string that starts every top-level conversation object
# in the ChatGPT chat.html export format.
_CONV_SENTINEL = b'{"async_status":'

# Fallback for older export formats that start differently
_CONV_SENTINEL_ALT = b'{"title":'


class ConversationIndex:
    """
    Holds the byte-offset index into the mmap'd file.
    Each entry is (title, start_offset, end_offset).
    Built once in a background thread.
    """
    def __init__(self, mm: mmap.mmap, file_size: int):
        self.mm = mm
        self.file_size = file_size
        self.entries: List[Tuple[str, int, int]] = []

    def build(self) -> None:
        """
        Scan the mmap for conversation sentinels and record offsets.
        Extracts the title from a lightweight regex scan rather than
        full JSON parsing, so this completes in ~1-2 seconds even on
        a 342 MB file.
        """
        sentinel = _CONV_SENTINEL
        positions: List[int] = []
        pos = 0
        while True:
            found = self.mm.find(sentinel, pos)
            if found == -1:
                break
            positions.append(found)
            pos = found + 1

        if not positions:
            # Try alternate sentinel
            sentinel = _CONV_SENTINEL_ALT
            pos = 0
            while True:
                found = self.mm.find(sentinel, pos)
                if found == -1:
                    break
                positions.append(found)
                pos = found + 1

        if not positions:
            return

        # Title regex: matches "title": "..." inside each conversation's JSON.
        # The title field lives at the END of each object (after the full mapping
        # dict), so it can be 2 KB – 480 KB into the object. We must scan the
        # full slice. mmap slices are zero-copy views so this is still fast.
        title_re = re.compile(rb'"title"\s*:\s*"((?:[^"\\]|\\.)*)"')

        for i, start in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else self.file_size
            window = self.mm[start:end]
            # findall gives every match; take the last one to prefer the
            # top-level title over any nested message attachment titles.
            matches = title_re.findall(window)
            if matches:
                raw_title = matches[-1]
                try:
                    title = json.loads(b'"' + raw_title + b'"')
                except (json.JSONDecodeError, ValueError):
                    title = raw_title.decode("utf-8", errors="replace")
                title = title.strip() or f"Conversation {i + 1}"
            else:
                title = f"Conversation {i + 1}"
            self.entries.append((title, start, end))


class MessageParser:
    """
    Parses a single conversation's JSON blob (already sliced from the mmap)
    into a flat ordered list of (role, text) message tuples.
    Uses the tree-traversal logic from ChatGPTFrictionHarvester for
    consistent ordering across both the viewer and the harvester.
    """

    @staticmethod
    def parse(raw_bytes: bytes) -> List[Tuple[str, str]]:
        """
        Returns a list of (role, text) pairs in chronological order.
        Handles the branching mapping tree by following the current_node
        path back through parent links when possible, or by doing a
        DFS from nodes with no parent otherwise.
        """
        raw = raw_bytes.rstrip(b", \n\r")
        try:
            conv = json.loads(raw)
        except json.JSONDecodeError:
            # Try stripping a trailing comma that separates array elements
            try:
                conv = json.loads(raw.rstrip(b","))
            except json.JSONDecodeError:
                return [("system", "[Could not parse conversation JSON]")]

        mapping = conv.get("mapping", {})
        if not mapping:
            return []

        # Build a lookup of id -> node
        nodes = {nid: node for nid, node in mapping.items()}

        # Find root nodes (nodes with no parent or whose parent is None/missing)
        # and traverse depth-first to produce an ordered message list
        children_map: dict = {}
        root_ids: List[str] = []
        for nid, node in nodes.items():
            parent_id = node.get("parent")
            if parent_id is None or parent_id not in nodes:
                root_ids.append(nid)
            else:
                children_map.setdefault(parent_id, []).append(nid)

        messages: List[Tuple[str, str]] = []

        def dfs(node_id: str) -> None:
            node = nodes.get(node_id)
            if not node:
                return
            msg = node.get("message")
            if msg:
                role = msg.get("author", {}).get("role", "unknown")
                parts = msg.get("content", {}).get("parts", [])
                text = "\n".join(str(p) for p in parts if isinstance(p, str)).strip()
                if text and role in ("user", "assistant", "system"):
                    messages.append((role, text))
            for child_id in children_map.get(node_id, []):
                dfs(child_id)

        for rid in root_ids:
            dfs(rid)

        return messages


class FrictionTagger:
    """
    Lightweight version of the ChatGPTFrictionHarvester tag logic,
    extracted so the viewer can apply visual tags without importing
    torch or triggering the full harvester init.
    """
    CREATOR_ALIASES = ["ila", "akkaris", "willabusta"]
    AI_KEYWORDS = ["archetype", "entity", "system", "architecture", "non-human", "ai"]
    ROLEPLAY_PATTERNS = [
        "i am ", "i will act as ", "playing the role of ", "persona: ",
        "act as ", "you are a ", "pretend to be ", "assume the role ",
        "imagine you are ", "roleplay ", "in the style of ", "respond as ",
        "take on the persona ", "you will be ", "portraying ", "simulating ",
        "acting as ", "representing ", "emulating ", "personifying ",
        "impersonating ", "channeling ",
    ]

    def tag(self, text: str, prev_tokens: set, prev_len: int) -> dict:
        text_lower = text.lower()
        tags: dict = {}

        if any(a in text_lower for a in self.CREATOR_ALIASES):
            tags["is_human_alias"] = 1.0
        if any(k in text_lower for k in self.AI_KEYWORDS):
            tags["is_nonhuman_archetype"] = 1.0
        if any(p in text_lower for p in self.ROLEPLAY_PATTERNS):
            tags["is_character_play"] = 1.0

        if "is_human_alias" in tags and "is_nonhuman_archetype" in tags:
            tags["veto_status"] = "saturation_escalation"

        # Jaccard shift
        current_tokens = set(text_lower.split())
        if prev_tokens and current_tokens:
            inter = len(prev_tokens & current_tokens)
            union = len(prev_tokens | current_tokens)
            jaccard = inter / max(1, union)
            if jaccard < 0.05 and len(current_tokens) > 10:
                tags["jarring_shift"] = 1.0

        # Dead-end cliff
        if prev_len > 500 and len(text) < 50 and not tags.get("is_character_play"):
            tags["dead_end_cliff"] = 1.0

        return tags


class FastChatViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fast Chat Viewer")
        self.geometry("1100x800")

        self.filepath: Optional[str] = None
        self.index: Optional[ConversationIndex] = None
        self.mmap_obj: Optional[mmap.mmap] = None
        self.file_obj = None
        self.tagger = FrictionTagger()

        # Render cancellation token: increment to cancel any in-flight render
        self._render_gen = 0

        # Pagination state
        self.current_page = 0
        self.page_size = 200
        self._filtered_entries = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="Open chat.html", command=self.open_file).pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(toolbar, text="Ready.")
        self.lbl_status.pack(side=tk.LEFT, padx=10)

        self.progress = ttk.Progressbar(toolbar, mode="indeterminate", length=200)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))

        # Left: conversation list with search
        left_outer = ttk.Frame(paned)
        paned.add(left_outer, weight=1)

        search_frame = ttk.Frame(left_outer)
        search_frame.pack(fill=tk.X)
        ttk.Label(search_frame, text="Filter:").pack(side=tk.LEFT)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self._on_search)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Frame for Listbox and scrollbar
        list_frame = ttk.Frame(left_outer)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=2)

        self.listbox = tk.Listbox(list_frame, font=("Segoe UI", 10), activestyle="dotbox")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_conv)

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=sb.set)

        # Pagination controls frame
        page_frame = ttk.Frame(left_outer)
        page_frame.pack(fill=tk.X, pady=4)

        self.btn_prev = ttk.Button(page_frame, text="Prev", command=self.prev_page, width=6)
        self.btn_prev.pack(side=tk.LEFT, padx=2)

        self.lbl_page = ttk.Label(page_frame, text="Page 1 of 1 (0 total)")
        self.lbl_page.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor=tk.CENTER)

        self.btn_next = ttk.Button(page_frame, text="Next", command=self.next_page, width=6)
        self.btn_next.pack(side=tk.RIGHT, padx=2)

        # Right: message text area
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)

        self.text_area = tk.Text(
            right_frame,
            wrap=tk.WORD,
            font=("Consolas", 11),
            state=tk.DISABLED,
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb2 = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=sb2.set)

        # Text tags
        self.text_area.tag_config("role_user",      foreground="#1a73e8", font=("Consolas", 10, "bold"))
        self.text_area.tag_config("role_assistant",  foreground="#137333", font=("Consolas", 10, "bold"))
        self.text_area.tag_config("role_system",     foreground="#888888", font=("Consolas", 10, "italic"))
        self.text_area.tag_config("dead_end",        background="#ffcccc")
        self.text_area.tag_config("jarring_shift",   background="#fff3cc")
        self.text_area.tag_config("character_play",  background="#ccffcc")
        self.text_area.tag_config("human_alias",     background="#e8d0ff")
        self.text_area.tag_config("separator",       foreground="#cccccc")

        # Filtered index mapping listbox row -> conversations index entry
        self._filtered: List[int] = []

    # ------------------------------------------------------------------
    # File Open & Indexing
    # ------------------------------------------------------------------

    def open_file(self) -> None:
        initial_dir = (
            r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner"
            r"\data\raw\chatgpt_userIla_and_archetpes_dyads_data"
        )
        if not os.path.exists(initial_dir):
            initial_dir = "/"
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select chat.html",
            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")],
        )
        if not path:
            return

        self.filepath = path
        self._render_gen += 1
        self._set_status(f"Indexing {os.path.basename(path)}...", busy=True)
        self._clear_list()

        threading.Thread(target=self._index_worker, daemon=True).start()

    def _index_worker(self) -> None:
        try:
            if self.mmap_obj:
                try:
                    self.mmap_obj.close()
                except Exception:
                    pass
            if self.file_obj:
                try:
                    self.file_obj.close()
                except Exception:
                    pass

            self.file_obj = open(self.filepath, "rb")
            self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            file_size = len(self.mmap_obj)

            idx = ConversationIndex(self.mmap_obj, file_size)
            idx.build()

            self.after(0, self._indexing_done, idx)
        except Exception as e:
            self.after(0, self._set_status, f"Indexing error: {e}", False)

    def _indexing_done(self, idx: ConversationIndex) -> None:
        self.index = idx
        self._set_status(f"Indexed {len(idx.entries)} conversations.", busy=False)
        self._apply_filter_and_reset_page()

    def _apply_filter_and_reset_page(self) -> None:
        self.current_page = 0
        self._filtered_entries = []
        if not self.index:
            self._rebuild_list()
            return
        
        flt = self.search_var.get().lower().strip()
        for i, entry in enumerate(self.index.entries):
            title = entry[0]
            if not flt or flt in title.lower():
                self._filtered_entries.append((i, entry))
        self._rebuild_list()

    def _rebuild_list(self) -> None:
        self.listbox.delete(0, tk.END)
        self._filtered = []
        
        total_items = len(self._filtered_entries)
        total_pages = max(1, (total_items + self.page_size - 1) // self.page_size)
        
        # Clamp page index
        if self.current_page >= total_pages:
            self.current_page = total_pages - 1
        if self.current_page < 0:
            self.current_page = 0
            
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, total_items)
        
        for idx in range(start_idx, end_idx):
            original_idx, (title, _, _) = self._filtered_entries[idx]
            self.listbox.insert(tk.END, title)
            self._filtered.append(original_idx)
            
        # Update page label and buttons
        self.lbl_page.config(text=f"Page {self.current_page + 1} of {total_pages} ({total_items} total)")
        self.btn_prev.config(state=tk.NORMAL if self.current_page > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_page < total_pages - 1 else tk.DISABLED)

    def prev_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self._rebuild_list()

    def next_page(self) -> None:
        total_items = len(self._filtered_entries)
        total_pages = max(1, (total_items + self.page_size - 1) // self.page_size)
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self._rebuild_list()

    def _on_search(self, *_) -> None:
        if self.index:
            self._apply_filter_and_reset_page()

    # ------------------------------------------------------------------
    # Conversation Selection & Loading
    # ------------------------------------------------------------------

    def _on_select_conv(self, _event) -> None:
        sel = self.listbox.curselection()
        if not sel or not self.index:
            return
        listbox_row = sel[0]
        if listbox_row >= len(self._filtered):
            return
        conv_idx = self._filtered[listbox_row]
        title, start, end = self.index.entries[conv_idx]

        self._render_gen += 1
        gen = self._render_gen

        self._set_text_content("")
        self._set_status(f"Loading: {title}", busy=True)

        threading.Thread(
            target=self._parse_worker,
            args=(start, end, gen),
            daemon=True,
        ).start()

    def _parse_worker(self, start: int, end: int, gen: int) -> None:
        if gen != self._render_gen:
            return
        try:
            self.mmap_obj.seek(start)
            raw = self.mmap_obj.read(end - start)
            messages = MessageParser.parse(raw)
            if gen == self._render_gen:
                self.after(0, self._render_messages, messages, gen)
        except Exception as e:
            if gen == self._render_gen:
                self.after(0, self._set_status, f"Parse error: {e}", False)

    # ------------------------------------------------------------------
    # Incremental Rendering
    # ------------------------------------------------------------------

    def _render_messages(self, messages: List[Tuple[str, str]], gen: int) -> None:
        if gen != self._render_gen:
            return
        self._set_status(f"Rendering {len(messages)} messages...", busy=False)
        self._set_text_enabled(True)
        self.text_area.delete(1.0, tk.END)
        self._set_text_enabled(False)
        self._insert_batch(messages, 0, gen)

    def _insert_batch(self, messages: List[Tuple[str, str]], start: int, gen: int) -> None:
        """Insert up to BATCH_SIZE messages, then yield via after() for the next batch."""
        if gen != self._render_gen:
            return

        BATCH_SIZE = 10  # small batches keep the UI responsive between flushes

        end = min(start + BATCH_SIZE, len(messages))
        prev_tokens: set = set()
        prev_len: int = 0

        self._set_text_enabled(True)
        for i in range(start, end):
            role, text = messages[i]
            tags = self.tagger.tag(text, prev_tokens, prev_len)

            # Role header
            role_tag = f"role_{role}" if role in ("user", "assistant", "system") else "role_system"
            role_label = role.upper()
            self.text_area.insert(tk.END, f"{role_label}\n", role_tag)

            # Message body
            body_start = self.text_area.index(tk.INSERT)
            self.text_area.insert(tk.END, text + "\n")
            body_end = self.text_area.index(tk.INSERT)

            # Apply friction tags to body
            if tags.get("dead_end_cliff"):
                self.text_area.tag_add("dead_end", body_start, body_end)
            if tags.get("jarring_shift"):
                self.text_area.tag_add("jarring_shift", body_start, body_end)
            if tags.get("is_character_play"):
                self.text_area.tag_add("character_play", body_start, body_end)
            if tags.get("is_human_alias"):
                self.text_area.tag_add("human_alias", body_start, body_end)

            # Separator line
            self.text_area.insert(tk.END, "\u2500" * 60 + "\n", "separator")

            prev_tokens = set(text.lower().split())
            prev_len = len(text)

        self._set_text_enabled(False)

        if end < len(messages):
            self._set_status(f"Rendering... {end}/{len(messages)} messages", busy=False)
            # 5 ms delay between batches — keeps the event loop alive
            self.after(5, self._insert_batch, messages, end, gen)
        else:
            self._set_status(f"Done. {len(messages)} messages.", busy=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_text_enabled(self, enabled: bool) -> None:
        self.text_area.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _set_text_content(self, text: str) -> None:
        self._set_text_enabled(True)
        self.text_area.delete(1.0, tk.END)
        if text:
            self.text_area.insert(tk.END, text)
        self._set_text_enabled(False)

    def _clear_list(self) -> None:
        self.listbox.delete(0, tk.END)
        self._filtered = []

    def _set_status(self, msg: str, busy: bool = False) -> None:
        self.lbl_status.config(text=msg)
        if busy:
            if not self.progress.winfo_ismapped():
                self.progress.pack(side=tk.LEFT, padx=5)
            self.progress.start(10)
        else:
            self.progress.stop()
            if self.progress.winfo_ismapped():
                self.progress.pack_forget()

    def on_closing(self) -> None:
        self._render_gen += 1
        for obj in (self.mmap_obj, self.file_obj):
            if obj:
                try:
                    obj.close()
                except Exception:
                    pass
        self.destroy()


if __name__ == "__main__":
    app = FastChatViewer()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
