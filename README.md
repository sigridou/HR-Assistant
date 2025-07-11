﻿# HR-Assistant

A local AI-powered assistant for querying HR documents (PDFs and JSON datasets) using a document-aware LLM.

---

## 📦 Installation

1. **Clone the repo / download source files**
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start your Ollama server:

```bash
ollama serve
```

Make sure you have the embedding model installed:

```bash
ollama pull mxbai-embed-large:335m
```

---

## 🚀 How to Use It

1. **Run the program**

```bash
python main.py
```

2. A GUI window will appear.

3. **Set your desired chunk size in characters** — this controls how the document or JSON records are split for embedding similarity search.
   - If your document has long paragraphs or you want broader context per chunk, use a higher value (e.g. 300).
   - If your data is highly structured (like small JSON records), use a smaller chunk size (e.g. 100–200).

4. **Click Upload PDF** to load an HR document file.

5. Or **Upload JSON Database** to load a structured HR data file (must be a JSON array of objects).

6. The system will convert and split the data into chunks of your chosen size, embed them locally, and build a searchable context vault.

7. **Type your HR-related question into the query box** and click **Ask Assistant**.

8. The assistant will reply using only information from the uploaded documents.

9. You’re free to query it as often as you like — upload more files at any time.

---

## ✅ Notes
- No internet APIs used. All processing is local via Ollama server.
- Ensure `ollama serve` is running before starting the app.
- Embedding model: `mxbai-embed-large:335m` (pre-pulled locally).
- Answers come strictly from your uploaded document vault context.

---

## 📂 Output Storage
- Processed text chunks are saved to `vault/vault.txt`
- New uploads append to this file.
- Embeddings regenerated every time you upload.

---

## 📌 Requirements
- Python 3.9+
- ollama
- torch
- PyPDF2
- tkinter
