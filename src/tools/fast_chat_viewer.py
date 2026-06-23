"""
Fast Chat Viewer.
A Tkinter-based lazy-loading viewer for massive ChatGPT `chat.html` exports.
Prevents browser freezing on 350MB+ HTML files by using memory-mapped reading
and regex-based block extraction.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import mmap
import re
import os

class FastChatViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fast Chat Viewer (Large HTML Virtualizer)")
        self.geometry("1000x800")
        
        self.filepath = None
        self.conversations = [] # List of (title, start_pos, end_pos)
        self.mmap_obj = None
        self.file_obj = None
        self.current_render_id = 0
        
        self._build_ui()
        
    def _build_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        btn_open = ttk.Button(toolbar, text="Open chat.html", command=self.open_file)
        btn_open.pack(side=tk.LEFT)
        
        self.lbl_status = ttk.Label(toolbar, text="Ready.")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
        self.progress = ttk.Progressbar(toolbar, mode='indeterminate', length=150)
        
        # Paned Window
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left Panel: Conversation List
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        self.listbox = tk.Listbox(left_frame, font=("Segoe UI", 10))
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_conv)
        
        scroll_y = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scroll_y.set)
        
        # Right Panel: Content Text
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        self.text_area = tk.Text(right_frame, wrap=tk.WORD, font=("Consolas", 11))
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scroll_t = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        scroll_t.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=scroll_t.set)
        
        # Setup tags for archetypes
        self.text_area.tag_config("dead_end", background="#ffcccc", foreground="black", font=("Consolas", 11, "bold"))
        self.text_area.tag_config("jarring_shift", background="#ffffcc", foreground="black", font=("Consolas", 11, "bold"))
        self.text_area.tag_config("character_play", background="#ccffcc", foreground="black", font=("Consolas", 11, "bold"))

    def open_file(self):
        initial_dir = r"d:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\raw\chatgpt_userIla_and_archetpes_dyads_data"
        if not os.path.exists(initial_dir):
            initial_dir = "/"
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select chat.html",
            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")]
        )
        if not path:
            return
            
        self.filepath = path
        self.lbl_status.config(text=f"Indexing {os.path.basename(path)}... Please wait.")
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.start()
        
        # Cancel any active load/render
        self.current_render_id += 1
        
        self._index_file()

    def _index_file(self):
        """Memory maps the file and finds conversation boundaries without loading to RAM."""
        try:
            if self.mmap_obj:
                self.mmap_obj.close()
            if self.file_obj:
                self.file_obj.close()
                
            self.file_obj = open(self.filepath, "rb")
            self.mmap_obj = mmap.mmap(self.file_obj.fileno(), 0, access=mmap.ACCESS_READ)
            
            self.conversations = []
            self.listbox.delete(0, tk.END)
            
            # Start background thread for indexing to keep GUI responsive
            import threading
            threading.Thread(target=self._index_file_worker, daemon=True).start()
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Failed to index file:\n{e}")
            self.lbl_status.config(text="Error.")

    def _index_file_worker(self):
        try:
            pattern = re.compile(b'<h4>(.*?)</h4>', re.IGNORECASE)
            file_size = len(self.mmap_obj)
            chunk_size = 1024 * 1024 * 5 # 5MB virtual chunks
            
            markers = []
            for match in pattern.finditer(self.mmap_obj):
                title = match.group(1).decode('utf-8', errors='ignore')
                title = re.sub(r'<[^>]+>', '', title).strip()
                markers.append((title, match.start()))
                
            conversations = []
            if markers:
                for i in range(len(markers)):
                    title, start_pos = markers[i]
                    end_pos = markers[i+1][1] if i + 1 < len(markers) else file_size
                    conversations.append((title, start_pos, end_pos))
            else:
                num_chunks = (file_size // chunk_size) + 1
                for i in range(num_chunks):
                    start_pos = i * chunk_size
                    end_pos = min((i + 1) * chunk_size, file_size)
                    title = f"Chunk {i+1} ({start_pos//1024}KB - {end_pos//1024}KB)"
                    conversations.append((title, start_pos, end_pos))
                    
            self.after(0, self._indexing_done, conversations)
            
        except Exception as e:
            self.after(0, self._indexing_failed, str(e))

    def _indexing_done(self, conversations):
        self.progress.stop()
        self.progress.pack_forget()
        self.conversations = conversations
        self.listbox.delete(0, tk.END)
        for title, _, _ in self.conversations:
            self.listbox.insert(tk.END, title)
        self.lbl_status.config(text=f"Indexed {len(self.conversations)} sections.")
        
    def _indexing_failed(self, error_msg):
        self.progress.stop()
        self.progress.pack_forget()
        messagebox.showerror("Error", f"Failed to index file:\n{error_msg}")
        self.lbl_status.config(text="Error.")

    def on_select_conv(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
            
        idx = selection[0]
        title, start_pos, end_pos = self.conversations[idx]
        
        self._load_content(start_pos, end_pos)
        
    def _load_content(self, start_pos, end_pos):
        if not self.mmap_obj:
            return
            
        # Cancel any ongoing rendering
        self.current_render_id += 1
        render_id = self.current_render_id
        
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Loading...")
        
        self.lbl_status.config(text="Loading section data...")
        self.progress.pack(side=tk.LEFT, padx=5)
        self.progress.start()
        
        # Read the chunk
        self.mmap_obj.seek(start_pos)
        raw_bytes = self.mmap_obj.read(end_pos - start_pos)
        
        import threading
        threading.Thread(
            target=self._load_content_worker,
            args=(raw_bytes, render_id),
            daemon=True
        ).start()

    def _load_content_worker(self, raw_bytes, render_id):
        try:
            if render_id != self.current_render_id:
                return
                
            text = raw_bytes.decode('utf-8', errors='replace')
            blocks = re.split(r'<div', text)
            
            # Import harvester logic for tagging
            import sys
            import os
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
            from src.data.chatgpt_friction_harvester import ChatGPTFrictionHarvester
            harvester = ChatGPTFrictionHarvester(export_dir="")
            
            last_user_tokens = set()
            last_user_len = 0
            last_user_text = ""
            
            parsed_blocks = []
            
            for block in blocks:
                if render_id != self.current_render_id:
                    return
                    
                if not block.strip():
                    continue
                
                clean_block = block.replace("<br>", "\n").replace("</p>", "\n\n")
                clean_block = re.sub(r'<[^>]+>', '', "<div" + clean_block)
                clean_block = clean_block.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()
                
                if not clean_block:
                    continue
                    
                tags = harvester._extract_tags(clean_block)
                current_tokens = set(clean_block.lower().split())
                
                jaccard = 1.0
                if last_user_tokens and current_tokens:
                    intersection = len(last_user_tokens.intersection(current_tokens))
                    union = len(last_user_tokens.union(current_tokens))
                    jaccard = intersection / max(1, union)
                    
                if jaccard < 0.05 and len(current_tokens) > 10:
                    tags["jarring_shift"] = 1.0
                    
                if last_user_len > 500 and len(clean_block) < 50:
                    if not tags.get("is_character_play"):
                        tags["dead_end_cliff"] = 1.0
                        
                if len(clean_block) > 50:
                    last_user_tokens = current_tokens
                    last_user_len = len(clean_block)
                    last_user_text = clean_block
                    
                parsed_blocks.append((clean_block, tags))
                
            if render_id != self.current_render_id:
                return
                
            self.after(0, self._render_content, parsed_blocks, render_id)
            
        except Exception as e:
            self.after(0, self._render_error, str(e), render_id)

    def _render_content(self, parsed_blocks, render_id):
        if render_id != self.current_render_id:
            return
        self.progress.stop()
        self.progress.pack_forget()
        self.text_area.delete(1.0, tk.END)
        self._insert_blocks_incremental(parsed_blocks, 0, render_id)
        
    def _render_error(self, error_msg, render_id):
        if render_id != self.current_render_id:
            return
        self.progress.stop()
        self.progress.pack_forget()
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, f"Error processing content:\n{error_msg}")
        self.lbl_status.config(text="Error.")

    def _insert_blocks_incremental(self, parsed_blocks, start_idx, render_id):
        if render_id != self.current_render_id:
            return
            
        batch_size = 50
        end_idx = min(start_idx + batch_size, len(parsed_blocks))
        
        for i in range(start_idx, end_idx):
            clean_block, tags = parsed_blocks[i]
            
            start_index = self.text_area.index(tk.INSERT)
            self.text_area.insert(tk.END, clean_block + "\n\n")
            end_index = self.text_area.index(tk.INSERT)
            
            if tags.get("dead_end_cliff"):
                self.text_area.tag_add("dead_end", start_index, end_index)
            if tags.get("jarring_shift"):
                self.text_area.tag_add("jarring_shift", start_index, end_index)
            if tags.get("is_character_play"):
                self.text_area.tag_add("character_play", start_index, end_index)
                
        if end_idx < len(parsed_blocks):
            self.lbl_status.config(text=f"Loaded {end_idx}/{len(parsed_blocks)} blocks...")
            self.after(10, self._insert_blocks_incremental, parsed_blocks, end_idx, render_id)
        else:
            self.lbl_status.config(text=f"Load complete. Loaded {len(parsed_blocks)} blocks.")

    def on_closing(self):
        self.current_render_id += 1
        if self.mmap_obj:
            try:
                self.mmap_obj.close()
            except:
                pass
        if self.file_obj:
            try:
                self.file_obj.close()
            except:
                pass
        self.destroy()

if __name__ == "__main__":
    app = FastChatViewer()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
