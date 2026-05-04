# 📘 Cambridge vs GPT Evaluation System

## 🔍 Overview

This project presents a **comparative evaluation framework** for automated grading systems by comparing:

* **GPT-based grading** (semantic understanding)
* **RAG-based grading** (retrieval + rubric grounding)

The system uses **Cambridge O Level Computer Science past papers and mark schemes** to evaluate how well AI grading aligns with official examination standards.

---

## 🎯 Objectives

* Evaluate student answers using **GPT and RAG**
* Compare grading outputs against **Cambridge marking schemes**
* Measure performance using:

  * BLEU
  * ROUGE
  * BERTScore
  * Mark Accuracy
* Build a **retrieval-based evaluation pipeline using FAISS**

---

## 🧩 Project Structure

```
cambridge_cs_evaluator/
│
├── data/
│   ├── question_papers/        # Cambridge question papers (PDF)
│   ├── mark_schemes/           # Corresponding mark schemes (PDF)
│
├── ingestion/
│   ├── extract.py              # PDF text extraction using pdfplumber
│   ├── question_splitter.py    # Splits papers into individual questions
│   ├── chunker.py              # Splits text into chunks for embedding
│   ├── pair_builder.py         # Matches QP with MS and builds dataset
│
├── rag/
│   ├── embedder.py             # Generates embeddings using OpenAI API
│   ├── store.py                # FAISS index creation and retrieval
│
├── evaluation/
│   ├── gpt_grader.py           # GPT-based grading
│   ├── rag_grader.py           # RAG-based rubric grading
│   ├── compare.py              # Compares outputs and prints analysis
│
├── app.py                      # Main interactive evaluation app
├── build_index.py              # One-time FAISS index builder
├── config.py                   # API keys and model configuration
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

### 1. Data Ingestion

* Extracts text from PDFs
* Splits into individual questions
* Pairs questions with mark schemes

### 2. Embedding & Storage

* Converts text into embeddings
* Stores vectors using **FAISS**

### 3. Retrieval (RAG)

* Finds most relevant mark scheme chunks
* Provides context for grading

### 4. Evaluation

* **GPT Grader** → semantic evaluation
* **RAG Grader** → strict rubric-based evaluation

### 5. Comparison

* Outputs grading differences
* Computes NLP metrics

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd cambridge_cs_evaluator
```

### 2. Create Virtual Environment

```bash
python -m venv rag_env
rag_env\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API

Create a `.env` or edit `config.py`:

```
OPENAI_API_KEY=your_key_here
```

---

## 🏗️ Build FAISS Index (Run Once)

```bash
python build_index.py
```

This step:

* Processes all PDFs
* Creates embeddings
* Stores FAISS index locally

---

## ▶️ Run Application

```bash
python app.py
```

The system will:

1. Ask a random Cambridge question
2. Take student input
3. Evaluate using GPT and RAG
4. Display marks + comparison
5. Show BLEU, ROUGE, BERTScore

---

## 📊 Evaluation Metrics

| Metric    | Purpose                    |
| --------- | -------------------------- |
| BLEU      | Measures n-gram overlap    |
| ROUGE     | Measures recall similarity |
| BERTScore | Semantic similarity        |
| Accuracy  | Marks alignment            |

---

## ⚠️ Notes

* First-time indexing may take time due to embedding generation
* Requires stable internet connection (OpenAI API)
* FAISS index is stored locally for faster reuse

---

## 📌 Future Improvements

* Better question extraction (remove instructions noise)
* Offline embedding support
* Fine-tuned grading model
* GUI interface (Streamlit)

---

## 👩‍💻 Authors

* Faryal Shamsi

---

## 📜 License

This project is for academic and research purposes.
