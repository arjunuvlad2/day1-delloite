# AI Solution Architect Program - Day 1 Labs

**Instructor: Nagarjuna Naidu**

Hands-on notebooks for **Day 1: AI Fundamentals & Data Foundations for Architects**.
You will work through three labs alongside the instructor. Each notebook has short "Your turn" exercises with hints; the answers are shown by the instructor in class.

| Lab | Notebook | When | What you build |
|---|---|---|---|
| 1 | `lab1_ai_lifecycle.ipynb` | Morning and early afternoon | Three flavours of ML, then the full AI lifecycle (CRISP-DM + MLOps) on a customer-churn case: cost matrix, leakage, pipeline, models, threshold, deployment, drift monitoring; plus "when NOT to use AI" |
| 2 | `lab2_data_foundations.ipynb` | Mid afternoon | Data types, batch and streaming pipelines, feature engineering, data-quality checks, labeling, PII masking, data architecture |
| 3 | `lab3_search_unstructured.ipynb` | Late afternoon | An enterprise search system over 48 messy company documents: BM25, embeddings, hybrid search, evaluation (Recall@k, MRR, nDCG) |

## Setup (about 5 minutes)

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                          # then open a notebook
```

Always start Jupyter from **this folder**, so the notebooks can find `data/`.

### Before class (recommended)
Open `lab3_search_unstructured.ipynb` and run the first few cells. The first run downloads a small embedding model (about 130 MB) and caches it. If the download is blocked on your network, the notebook falls back automatically to a lightweight approach so you can still follow along.

## Tips
- Run cells with **Shift + Enter**, top to bottom. If something breaks, use *Kernel > Restart & Run All* to get a clean state.
- Try each **Your turn** cell before the instructor shows the answer: a couple of minutes of struggle teaches more than reading.
- The labs are longer than the time we have in the room. Use the unfinished parts as homework; they include stretch tasks.
- Generated files (models, charts, caches) are written to an `artifacts/` folder next to the notebooks. It is safe to delete.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate your virtual environment and re-run `pip install -r requirements.txt` |
| Notebook says `data/corpus` not found | Start Jupyter from the folder that contains the `data/` directory |
| Model download fails (Lab 3) | Continue: the notebook switches to its offline fallback and prints a warning |
| Plots do not appear | Run the first cell of the notebook, and make sure you are using the Jupyter kernel from your virtual environment |

All data used in these labs is synthetic. Company names, documents and customers are fictional.
