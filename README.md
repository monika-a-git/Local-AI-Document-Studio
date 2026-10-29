# ⚡ Local AI Document Studio

> **A professional, 100% private AI document intelligence and creative generation studio.**  
> Powered by **Ollama (`llama3.2:3b`, `nomic-embed-text`)**, **ChromaDB**, **LangChain**, **PyPDF**, **Pillow**, **ReportLab**, and **python-docx**.  
> Specially tuned for **Windows**, **CPU-only execution**, and **16 GB RAM**.

---

## 🌟 Key Features

### 1. 💬 Conversational RAG with Context-Aware Queries
- **Follow-up Query Contextualization:** Automatically reformulates follow-up queries (e.g., *"What are its voltage specs?"*) into standalone search queries using conversation history.
- **Hybrid Retrieval & Reranking:** Fuses dense vector semantic similarity with BM25 lexical keyword scoring to eliminate irrelevant chunk retrieval.
- **Source-Grounded Answers:** Strict prompt grounding ensures the model cites evidence with exact `[Source: filename, Page: X]` tags.
- **Evidence-Based Abstention:** If document evidence is missing or below relevance threshold, the assistant politely abstains rather than hallucinating.
- **Multi-Scope Search:** Query all documents or restrict retrieval to specific selected PDFs.
- **Chat History & Export:** Export transcripts to Markdown or TXT; clear chat memory at any time.

### 2. 📁 Multi-PDF Ingestion & Document Management
- **Incremental Indexing:** Ingest new PDFs without re-indexing existing documents.
- **SHA-256 Duplicate Detection:** Prevents re-indexing duplicate files even if renamed.
- **Accurate 1-Indexed Page Tracking:** Human-readable page numbers matching real PDF pages.
- **Document-Specific Deletion:** Safely removes specific files, registry entries, and their associated vectors from ChromaDB.
- **Memory Safety:** Validates file size (50MB threshold), prevents corrupt/encrypted PDF crashes, and protects system RAM.

### 3. ✨ AI Creative Studio & Visual Generation
Transform PDF knowledge into tailored studio assets:
- **Posters:** Academic Conference Posters (4-quadrant layout), Event Announcement Posters, Infographic Summaries, Educational Awareness Posters.
- **Brochures:** 1-Page Corporate Flyers, Multi-Page Booklets, Tri-Fold 6-Panel Brochures.
- **Presentations & Diagrams:** 8-Slide Presentation Decks with speaker notes, Process Flowcharts (Mermaid syntax), Component Architecture Diagrams, Chronological Milestone Timelines.
- **Data & Tables:** Structured Markdown/CSV extracted tables, Quantitative Summaries, Visual Chart datasets.
- **CPU-Friendly Offline Visuals:** Uses **Pillow** and **matplotlib** for high-resolution poster, brochure, and chart graphics (100% local, zero external GPU or image API costs).
- **Interactive Workspace:** Edit generated drafts in an interactive Markdown editor before exporting.

### 4. 📑 Reports Studio
Generate publication-ready formal documents:
- **Executive Summaries**
- **Academic Research Syntheses**
- **Technical System Documentation**
- **Study Notes & Review Guides**
- **Project Proposals**

### 5. 📥 Multi-Format Professional Exports
Download generated content in any standard format:
- **PDF (`.pdf`):** Formatted with ReportLab, including custom header bands, styled headings, bullet points, tables, and dynamic "Page X of Y" footers.
- **Word (`.docx`):** Styled Microsoft Word document via `python-docx` with headings, bullet lists, and tables.
- **Markdown (`.md`):** Clean markdown with frontmatter metadata.
- **Plain Text (`.txt`):** Structured ASCII text.
- **CSV (`.csv`):** Extracted tables exported to spreadsheet CSV.
- **PNG (`.png`):** High-resolution visual posters, brochures, and charts rendered via Pillow/matplotlib.

### 6. 📊 Real RAG Evaluation Suite
- **Indexing Throughput Benchmark:** Measures characters per second, chunk count, and parse duration.
- **Retrieval Quality Testing:** Measures latency, similarity score distribution, and ground-truth keyword presence.
- **Reciprocal Rank (MRR):** Calculates exact rank of first relevant chunk.
- **Automated Benchmark Suite:** Runs multi-query evaluations and exports results to CSV with **zero fabricated scores**.

---

## 🏗️ Architecture & Code Organization

```
Local Rag/
├── app.py                      # Modern Streamlit Dashboard (Chat, Docs, Create, Reports, Eval)
├── requirements.txt            # Complete project dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
│
├── src/                        # Reusable modular backend
│   ├── config.py               # Paths, models, chunk parameters, palettes, limits
│   ├── ollama_client.py        # Health checks, model verification & diagnostics
│   ├── pdf_processor.py        # PDF validation, hash detection, extraction & chunking
│   ├── document_manager.py     # Document registry, duplicate detection & safe deletion
│   ├── vector_store.py         # ChromaDB persistence, filtered retrieval & chunk deletion
│   ├── retrieval.py            # Hybrid retrieval (BM25 + vector fusion) & thresholding
│   ├── rag_pipeline.py         # Conversational RAG, follow-up query rewriting & citations
│   ├── content_generator.py    # Multi-format transformation studio & prompt templates
│   ├── visual_builder.py       # Pillow & matplotlib CPU visual rendering engine
│   ├── export_manager.py       # ReportLab PDF, python-docx, MD, TXT, CSV, PNG exports
│   └── evaluation.py           # Evaluation metrics (MRR, hit rate, latency, benchmarks)
│
├── data/
│   ├── uploads/                # Local PDF storage directory
│   ├── exports/                # Exported documents cache
│   └── document_registry.json  # Persistent document metadata registry
│
├── chroma_db/                  # Persistent ChromaDB vector database
└── tests/                      # Automated pytest suite (34 passing tests)
    ├── test_pdf_processor.py
    ├── test_document_manager.py
    ├── test_retrieval_and_rag.py
    ├── test_vector_store_and_generator.py
    ├── test_visual_and_exports.py
    ├── test_evaluation.py
    └── test_rag.py
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.11+**
- **Ollama** installed on your system ([Download Ollama](https://ollama.com/download))

### 2. Pull Required Models
Run in your terminal:
```bash
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Studio Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the full automated test suite covering all modules:
```bash
pytest tests/ -v
```
All 34 test cases validate PDF processing, duplicate detection, vector indexing, metadata filtering, citations, follow-up rewriting, visual generation, multi-format exports, and evaluation metrics.

---

## 🔒 Privacy & Performance Guarantees
- **100% Local Inference:** No data or embeddings leave your computer.
- **CPU Optimized:** Tuned chunk size (800 chars / 150 overlap) and lightweight visual generation minimize CPU load.
- **Memory Footprint:** Tested to run smoothly within a 16 GB RAM budget.

