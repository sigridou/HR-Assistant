import os
import PyPDF2
import re

def convert_pdf_to_text(max_length):
    from tkinter import filedialog
    file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if file_path:
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ' '.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = text.split('.')
        chunks, current_chunk = [], ''

        for sentence in sentences:
            sentence = sentence.strip()
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                padding = ' ' * (max_length - len(current_chunk))
                chunks.append(current_chunk + padding)
                current_chunk = sentence

        if current_chunk:
            padding = ' ' * (max_length - len(current_chunk))
            chunks.append(current_chunk + padding)

        os.makedirs("vault", exist_ok=True)
        with open("vault/vault.txt", "a", encoding="utf-8") as vault_file:
            for chunk in chunks:
                vault_file.write(chunk + "\n")
        print(f"PDF content added to vault.txt as padded {max_length}-char chunks.")
