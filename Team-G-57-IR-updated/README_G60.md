# Information Retrieval System — Streamlit App

## Overview
End-to-end Information Retrieval system built with Streamlit, covering:
- Text preprocessing pipeline
- Inverted index construction
- Phrase query (Biword + Positional Index)
- Dictionary search (BST vs B-Tree)
- Tolerant retrieval (wildcards, spelling correction, edit distance, Soundex)

---

## Dependencies

Python 3.9+

Install all required packages:

```bash
pip install streamlit nltk scikit-learn pandas numpy
```

Download required NLTK data (run once):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
```

---

## Running the App

```bash
streamlit run IR_System_G60.py
```

Then open http://localhost:8501 in your browser.

---

## Dataset Format

The app accepts a `.txt` file where each line is a document in one of two formats:

**Format 1 (recommended):**
```
DOC001: Your document text here...
DOC002: Another document text here...
```

**Format 2 (free text — one document per line):**
```
Your first document text here.
Your second document text here.
```

A sample dataset (`sample_documents.txt`, 17 documents) is bundled and loaded automatically if no file is uploaded.

---

## Sections

| Section | Description |
|---------|-------------|
| A — Upload & View | Upload documents and browse the collection |
| B — Text Preprocessing | Tokenization, lowercasing, stop-word removal, stemming, lemmatization, inverted index |
| C — Phrase Query | Biword index vs positional index comparison |
| D — Dictionary Search | BST vs B-Tree performance benchmarks |
| E — Tolerant Retrieval | Wildcards, spelling correction, edit distance, Soundex |
| G — Inference | Discussion and conclusions |

---

## Files

```
G60_Submission/
├── IR_System_G60.py            # Main Streamlit application
├── sample_documents.txt        # Built-in 17-document dataset
├── IR_Assignment_Report_G60.pdf # Assignment report
├── Team_Contribution_G60.xlsx  # Team contribution sheet
└── README_G60.md               # This file
```

---

## Notes

- No backend scripts needed — the entire workflow runs through the Streamlit frontend.
- Upload your own `.txt` dataset via the sidebar, or use the built-in sample.
- All experimental results (BST vs B-Tree timing tables, stemming comparison) are generated live.
