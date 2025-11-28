Sanskrit Document Retrieval-Augmented Generation (RAG) System  
**Platform:** Google Colab (Backend) • HTML/CSS/JS (Frontend optional)  
**Author:** Vedant Butle  

---

Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed specifically for **Sanskrit document question answering**.  
It works fully on **CPU**, uses **FAISS** for similarity search, and **Flan-T5** for generating answers in Sanskrit.

Users can upload a PDF, embed the text into vector space, retrieve the most relevant chunks, and generate accurate Sanskrit responses based strictly on the document.

---

Features  
- PDF ingestion and preprocessing  
- Text chunking with overlap  
- Dense embeddings using **jinaai/jina-embeddings-v3**  
- Similarity search using **FAISS (CPU)**  
- MMR-based diversified retrieval  
- Answer generation using **google/flan-t5-large**  
- Simple Gradio frontend for querying  
- Fully CPU-compatible → no GPU required  

---

Project Workflow  
1. **Upload PDF** (Sanskrit or Sanskrit-related content)  
2. **Extract & clean textual data**  
3. **Chunk long text into manageable segments**  
4. **Generate embeddings for each chunk**  
5. **Build FAISS index**  
6. **Retrieve relevant chunks with MMR (Maximal Marginal Relevance)**  
7. **Generate final answer using Flan-T5**  
8. **Display result through Gradio UI**  

---

Installation (Colab)
Run these commands to install dependencies:

```python
!pip install faiss-cpu pypdf gradio transformers
```

---

Upload Your PDF

```python
from google.colab import files
uploaded = files.upload()
print(uploaded.keys())
```

Update this line after uploading:

```python
pdf_path = "/content/YourUploadedFile.pdf"
```

---

Code Structure (Simplified Overview)

Load PDF
```python
def load_pdf(path):
    ...
```

Create Chunks
```python
chunks = create_chunks(documents, chunk_size=450, overlap=80)
```

Generate Embeddings  
Using `jinaai/jina-embeddings-v3`:

```python
embed_text(text)
```

Build FAISS Index
```python
index = faiss.IndexFlatIP(dim)
index.add(all_embeddings)
```

Retrieve Chunks using MMR
```python
indices = mmr(query_emb, all_embeddings, k=5)
```

Generate Answer (Sanskrit)
```python
answer = gen_model.generate(...)
```

Launch Gradio App
```python
gr.Interface(...).launch()
```

---

Gradio Interface  
The UI accepts:

- **Question** (Sanskrit or English transliteration)
- **Top-k value** (number of retrieved chunks)

Output includes:

- Generated Sanskrit Answer  
- Retrieved Context Blocks  

---

Prompt Template (Generation)

The model is instructed to:

- Rely **only** on retrieved document context  
- Respond in **Sanskrit**  
- Decline when context is missing  

```
"दस्तावेजे उत्तरं न विद्यते"  
```

---

Retrieval Algorithm: MMR  
MMR increases diversity in retrieved chunks:

```
score = λ * similarity_to_query – (1 – λ) * similarity_to_selected
```

This avoids retrieving repeated or similar chunks.

---

Limitations  
- Handles Sanskrit texts only  
- Accuracy depends heavily on quality of the uploaded PDF  
- On CPU, embedding + generation may be slow  
- Long PDFs may require runtime upgrades in Colab  

---

Future Improvements  
- Add multilingual support  
- Switch to quantized models for faster CPU inference  
- Add a web-based frontend with proper API integration  
- Use advanced embedding models (LAION, BGE, Instructor XL)  

---

Contact  
**Developer:** Vedant Butle  
Feel free to connect for improvements, debugging, or extension of this RAG system 

