# AI Solution Architect Program: Hands-on Labs

**Instructor: Nagarjuna Naidu**

Labs for the AI Solution Architect program. Each day has its own folder. Open the notebooks in order and follow along with the instructor.

| Day | Folder | Labs |
|---|---|---|
| 1 | `day1/` | Lab 1a: three flavours of machine learning |
| 2 | `day2/` | 2A tokens, embeddings and semantic search · 2B ML system design and KPIs · 2C LLM internals · 2D prompt patterns · 2E RAG assistant |

(Later days are added as the program progresses.)

## Setup (about 5 minutes)

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd day2                              # or day1
jupyter lab
```

Start Jupyter from inside the day folder you are working on, so each notebook can find its `data/` folder and helper files.

### Before Day 2 (recommended, 10 minutes)
1. Open `day2/day2_lab2a_embeddings.ipynb` and run the first two cells. This downloads a small embedding model (about 130 MB) once. If it is blocked on your network, tell the instructor.
2. **Optional but helpful:** get a free LLM key so the prompt and RAG labs run live. Every lab also works in *offline demo mode* without one.
   - Groq (free tier): create a key, then `LLM_BASE_URL=https://api.groq.com/openai/v1`
   - Google AI Studio (free tier): create a key, then `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
   - Local, no key: install Ollama, pull a small model, then `LLM_BASE_URL=http://localhost:11434/v1`
   - Azure OpenAI (if your organisation provides it): set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and `LLM_MODEL` to your deployment name.

   Set `LLM_API_KEY` and `LLM_MODEL` too (see the first cell of Lab 2D). Never paste confidential company data into a public service.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate your virtual environment and run `pip install -r requirements.txt` again |
| Notebook cannot find `data/` or `llm_client` | Start Jupyter from inside the day folder (for example `day2/`) |
| Embedding model download fails | Tell the instructor; labs have fallbacks where possible |
| Weaviate section does not run | It is optional (and not supported on Windows); use the Chroma sections |

All data used in these labs is synthetic. Company names, documents and customers are fictional.
