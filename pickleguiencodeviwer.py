import os
import pickle
import torch
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

class DataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Gyroidic Sparse Covariance Flux Reasoner - Data Inspector")
        self.root.geometry("1000x700")

        # Initial target path
        self.base_path = r"D:\programming\python\Gyroidic Sparse Covariance Flux Reasoner\data\encodings"
        if not os.path.exists(self.base_path):
            self.base_path = os.getcwd()
        
        self.setup_ui()
        self.refresh_file_list()

    def setup_ui(self):
        # Top toolbar
        self.toolbar = ttk.Frame(self.root, padding="5")
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.toolbar, text="Select Directory", command=self.select_directory).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Refresh", command=self.refresh_file_list).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Copy to Clipboard", command=self.copy_to_clipboard).pack(side=tk.LEFT, padx=5)
        
        self.path_label = ttk.Label(self.toolbar, text=f"Path: {self.base_path}", font=('Helvetica', 9))
        self.path_label.pack(side=tk.LEFT, padx=10)

        # PanedWindow for adjustable ratio
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left Panel: File List
        self.left_frame = ttk.Frame(self.paned, padding="10")
        self.paned.add(self.left_frame, width=300)

        ttk.Label(self.left_frame, text="Available Files:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)
        
        # Frame for listbox and scrollbars
        list_frame = ttk.Frame(self.left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.file_listbox = tk.Listbox(list_frame, font=('Consolas', 9))
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Vertical scrollbar for listbox
        v_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=v_scroll.set)

        # Horizontal scrollbar for listbox (filename wraparound/view)
        h_scroll = ttk.Scrollbar(self.left_frame, orient=tk.HORIZONTAL, command=self.file_listbox.xview)
        h_scroll.pack(fill=tk.X)
        self.file_listbox.config(xscrollcommand=h_scroll.set)

        self.file_listbox.bind('<<ListboxSelect>>', self.on_file_select)

        # Right Panel: Data Content
        self.right_frame = ttk.Frame(self.paned, padding="10")
        self.paned.add(self.right_frame)

        ttk.Label(self.right_frame, text="Data Structure / Content Preview:", font=('Helvetica', 10, 'bold')).pack(anchor=tk.W)

        self.text_area = tk.Text(self.right_frame, wrap=tk.NONE, font=('Consolas', 10))
        self.text_area.pack(fill=tk.BOTH, expand=True)

        # Scrollbars for text area
        text_v_scroll = ttk.Scrollbar(self.right_frame, orient=tk.VERTICAL, command=self.text_area.yview)
        text_v_scroll.place(in_=self.text_area, relx=1.0, relheight=1.0, anchor=tk.NE)
        self.text_area.config(yscrollcommand=text_v_scroll.set)

        text_h_scroll = ttk.Scrollbar(self.right_frame, orient=tk.HORIZONTAL, command=self.text_area.xview)
        text_h_scroll.pack(fill=tk.X)
        self.text_area.config(xscrollcommand=text_h_scroll.set)

    def select_directory(self):
        new_path = filedialog.askdirectory(initialdir=self.base_path)
        if new_path:
            self.base_path = new_path
            self.path_label.config(text=f"Path: {self.base_path}")
            self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        if not os.path.exists(self.base_path):
            return

        try:
            # Include all files to be a "full file viewer"
            files = sorted([f for f in os.listdir(self.base_path)])
            for f in files:
                self.file_listbox.insert(tk.END, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not list directory:\n{e}")

    def copy_to_clipboard(self):
        try:
            content = self.text_area.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(content)
            messagebox.showinfo("Success", "Content copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not copy to clipboard:\n{e}")

    def on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if not selection:
            return

        filename = self.file_listbox.get(selection[0])
        filepath = os.path.join(self.base_path, filename)
        
        self.text_area.delete(1.0, tk.END)
        
        # Check file size before reading
        try:
            file_size = os.path.getsize(filepath)
            if file_size > 10 * 1024 * 1024: # 10MB limit for raw text
                if not filename.endswith(('.pt', '.pkl')):
                    self.text_area.insert(tk.END, f"File too large to preview ({file_size / 1024 / 1024:.2f} MB)")
                    return
        except:
            pass

        try:
            if filename.endswith('.pt'):
                # Load PyTorch file
                data = torch.load(filepath, map_location='cpu')
                self.display_data(data, filename)
            elif filename.endswith('.pkl'):
                # Load Pickle file
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                self.display_data(data, filename)
            else:
                # Try reading as text
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(10000) # Read first 10k chars
                        self.text_area.insert(tk.END, f"--- Raw Text Preview ({filename}) ---\n\n{content}")
                        if file_size > 10000:
                            self.text_area.insert(tk.END, "\n\n... [TRUNCATED] ...")
                except Exception as e:
                    self.text_area.insert(tk.END, f"Binary file or unreadable as text: {filename}\nError: {e}")
        except Exception as e:
            self.text_area.insert(tk.END, f"Error reading file: {e}")

    def display_data(self, data, filename):
        self.text_area.insert(tk.END, f"--- Object Inspector ({filename}) ---\n\n")
        if isinstance(data, torch.Tensor):
            info = f"Type: PyTorch Tensor\nShape: {data.shape}\nDevice: {data.device}\nDtype: {data.dtype}\n\nContent Preview:\n{data}"
        elif isinstance(data, dict):
            info = f"Type: Dictionary\nItems: {len(data)}\nKeys: {list(data.keys())}\n\nFull Content:\n{data}"
        elif isinstance(data, list):
            info = f"Type: List\nLength: {len(data)}\n\nPreview (First 5 items):\n{data[:5]}"
        else:
            info = f"Type: {type(data)}\n\nContent:\n{data}"
        
        self.text_area.insert(tk.END, info)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataViewer(root)
    root.mainloop()