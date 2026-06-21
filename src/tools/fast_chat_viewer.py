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
        
        self._build_ui()
        
    def _build_ui(self):
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        btn_open = ttk.Button(toolbar, text="Open chat.html", command=self.open_file)
        btn_open.pack(side=tk.LEFT)
        
        self.lbl_status = ttk.Label(toolbar, text="Ready.")
        self.lbl_status.pack(side=tk.LEFT, padx=10)
        
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

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select chat.html",
            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")]
        )
        if not path:
            return
            
        self.filepath = path
        self.lbl_status.config(text=f"Indexing {os.path.basename(path)}... Please wait.")
        self.update()
        
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
            
            # Simple heuristic: ChatGPT HTML exports usually have <div class="message"> or similar
            # Or headers like <h4>Conversation Name</h4>
            # We will just split it into manageable 1MB chunks if no clear marker is found,
            # or try to find <div class="conversation">
            
            self.conversations = []
            self.listbox.delete(0, tk.END)
            
            # Regex for titles (very heuristic, depends on chatgpt export format)
            # Usually <h4>Title</h4> or similar inside <div class="conversation">
            # We'll do a simple scan
            pattern = re.compile(b'<h4>(.*?)</h4>', re.IGNORECASE)
            
            pos = 0
            file_size = len(self.mmap_obj)
            chunk_size = 1024 * 1024 * 5 # 5MB virtual chunks
            
            # If we don't find proper markers, just virtualize by 5MB chunks
            markers = []
            for match in pattern.finditer(self.mmap_obj):
                title = match.group(1).decode('utf-8', errors='ignore')
                # Clean html tags
                title = re.sub(r'<[^>]+>', '', title).strip()
                markers.append((title, match.start()))
                
            if markers:
                for i in range(len(markers)):
                    title, start_pos = markers[i]
                    end_pos = markers[i+1][1] if i + 1 < len(markers) else file_size
                    self.conversations.append((title, start_pos, end_pos))
                    self.listbox.insert(tk.END, title)
            else:
                # Fallback: Virtualize by 5MB chunks
                num_chunks = (file_size // chunk_size) + 1
                for i in range(num_chunks):
                    start_pos = i * chunk_size
                    end_pos = min((i + 1) * chunk_size, file_size)
                    title = f"Chunk {i+1} ({start_pos//1024}KB - {end_pos//1024}KB)"
                    self.conversations.append((title, start_pos, end_pos))
                    self.listbox.insert(tk.END, title)
                    
            self.lbl_status.config(text=f"Indexed {len(self.conversations)} sections.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to index file:\n{e}")
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
            
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, "Loading...")
        self.update()
        
        # Read the chunk
        self.mmap_obj.seek(start_pos)
        raw_bytes = self.mmap_obj.read(end_pos - start_pos)
        text = raw_bytes.decode('utf-8', errors='replace')
        
        # Simple HTML tag stripping for readability (Optional, but helps performance)
        # Replacing <br> and <p> with newlines
        text = text.replace("<br>", "\n").replace("</p>", "\n\n")
        # Strip remaining tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Replace html entities
        text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, text)

    def on_closing(self):
        if self.mmap_obj:
            self.mmap_obj.close()
        if self.file_obj:
            self.file_obj.close()
        self.destroy()

if __name__ == "__main__":
    app = FastChatViewer()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
