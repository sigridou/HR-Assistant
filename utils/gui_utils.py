import os
import json
from tkinter import Tk, Button, filedialog, messagebox, Label, Entry, Text, Scrollbar, END
from utils import file_utils, embedding_utils, chat_utils
import ollama
from config import DEFAULT_MODEL

client_model = DEFAULT_MODEL
vault_embeddings_tensor = None
vault_content = []
conversation_history = []
system_message = "Analyse HR documents and queries. Always refer to the uploaded documents as your knowledge base. Only answer based on their content."

def launch_main_gui():
    global vault_embeddings_tensor, vault_content, conversation_history

    root = Tk()
    root.title("HR Data Assistant")
    root.geometry("620x680")

    def get_chunk_size():
        try:
            size = int(chunk_size_entry.get())
            if size <= 0:
                raise ValueError
            return size
        except ValueError:
            messagebox.showerror("Invalid Chunk Size", "Please enter a positive integer for chunk size.")
            return None

    def upload_pdf():
        size = get_chunk_size()
        if size is None:
            return
        file_utils.convert_pdf_to_text(size)
        reload_embeddings()
        messagebox.showinfo("Upload Complete", "PDF content added to vault.")

    def upload_json():
        size = get_chunk_size()
        if size is None:
            return
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as jf:
                records = json.load(jf)
                if isinstance(records, list):
                    os.makedirs("vault", exist_ok=True)
                    with open("vault/vault.txt", "a", encoding="utf-8") as vf:
                        for entry in records:
                            line = json.dumps(entry)
                            padding = ' ' * (size - len(line)) if len(line) < size else ''
                            vf.write(line[:size] + padding + "\n")
                    reload_embeddings()
                    messagebox.showinfo("Upload Complete", "JSON records added to vault.")
                else:
                    messagebox.showerror("Format Error", "JSON must be an array of objects.")

    def reload_embeddings():
        global vault_embeddings_tensor, vault_content
        if not os.path.exists("vault/vault.txt"):
            messagebox.showinfo("Vault Empty", "No documents uploaded yet.")
            return
        with open("vault/vault.txt", "r", encoding="utf-8") as vf:
            vault_content = vf.readlines()
        if not vault_content:
            messagebox.showinfo("Vault Empty", "No content found in vault.txt.")
        else:
            vault_embeddings_tensor = embedding_utils.generate_embeddings(vault_content)
            if vault_embeddings_tensor.nelement() == 0:
                messagebox.showwarning("Embedding Warning", "No embeddings generated. Check if your embedding model is running.")

    def ask_query():
        if vault_embeddings_tensor is None or vault_embeddings_tensor.nelement() == 0:
            messagebox.showwarning("Missing Data", "Please upload documents before querying.")
            return
        query = query_entry.get()
        if not query:
            messagebox.showwarning("Missing Input", "Please enter a query.")
            return
        response = chat_utils.ollama_chat(
            query,
            system_message,
            vault_embeddings_tensor,
            vault_content,
            client_model,
            conversation_history
        )
        chat_output.insert(END, f"\nYou: {query}\n\nAssistant: {response}\n")
        query_entry.delete(0, END)

    Label(root, text="Chunk Size (characters):").pack(pady=5)
    chunk_size_entry = Entry(root, width=10)
    chunk_size_entry.insert(0, "208")
    chunk_size_entry.pack(pady=5)

    Button(root, text="Upload PDF", command=upload_pdf, width=20).pack(pady=5)
    Button(root, text="Upload JSON Database", command=upload_json, width=20).pack(pady=5)

    Label(root, text="Enter HR Query:").pack(pady=5)
    query_entry = Entry(root, width=60)
    query_entry.pack(pady=5)

    Button(root, text="Ask Assistant", command=ask_query, width=20).pack(pady=5)

    chat_output = Text(root, wrap='word', height=25, width=75)
    chat_output.pack(pady=5)

    scrollbar = Scrollbar(root, command=chat_output.yview)
    scrollbar.pack(side='right', fill='y')
    chat_output['yscrollcommand'] = scrollbar.set

    root.mainloop()
