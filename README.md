# AI Solution Architect Program: Hands-on Labs

**Instructor: Nagarjuna Naidu**

Labs for the AI Solution Architect program. Each day has its own folder. Open the notebooks in order and follow along with the instructor.

| Day | Folder | What you do |
|---|---|---|
| 1 | `day1/` | Lab 1a: three flavours of machine learning |
| 2 | `day2/` | Lab 2A vectors and semantic search (local + Weaviate Cloud) · Lab 2B ML system design and KPIs · Lab 2C prompting and RAG · demo notebook: LLM internals (instructor-led) |
| 3 | `day3/` | Lab 3A RAG quality and grounding · Lab 3B Agentic RAG demo |
| 4 | `project/` | Project milestones M1 to M3: assistant with tools, MCP, evaluate / track / monitor |
| 5 | `project/` | Project milestones M4 and M5: secure and explain, demo and design (M6 is an optional extra on multi-agent patterns) |
| Capstone | `capstone/` | Day 5 case study: brief and response sheet; optional architecture canvas |

On Days 4 and 5 you build one project, the **Northwind Knowledge Assistant**, in five milestones. Open `project/README.md` first.

## Setup (about 5 minutes)

You need Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd project                           # or day1 / day2 / day3
jupyter lab
```

Start Jupyter from inside the day folder you are working on, so each notebook can find its `data/` folder and helper files.

Note: `requirements.txt` includes the Day 4 and Day 5 project packages (MLflow, SHAP, MCP). They are large, so install everything once, the night before Day 4, on a good connection.

### Before Day 2 (recommended, 15 minutes)
1. Open `day2/day2_lab2a_embeddings.ipynb` and run the first two cells. This downloads a small embedding model (about 130 MB) once. If it is blocked on your network, tell the instructor.
2. **Weaviate Cloud (free, for the hosted-vectorization part of Lab 2A):** sign up at console.weaviate.cloud with an email you are allowed to use, create a **free cluster**, and copy the cluster **REST URL** and an **API key**. You will paste them in Lab 2A, section 7 (see *Where to put your API keys* below). Use synthetic data only. If you skip this, Lab 2A still works with a local model.
3. **Optional but helpful:** get a free LLM key so the prompt and RAG labs run live. Every lab also works in *offline demo mode* without one.
   - Groq (free tier): create a key, then `LLM_BASE_URL=https://api.groq.com/openai/v1`
   - Google AI Studio (free tier): create a key, then `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`
   - Local, no key: install Ollama, pull a small model, then `LLM_BASE_URL=http://localhost:11434/v1`
   - Azure OpenAI (if your organisation provides it): set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, and `LLM_MODEL` to your deployment name.

   See *Where to put your API keys* below. Never paste confidential company data into a public service.

## Where to put your API keys (step by step)

You need keys only for two optional things. **Nothing breaks without them:** Lab 2A uses a local model and Lab 2C runs in offline demo mode.

| What | Variables to set | Used in |
|---|---|---|
| Weaviate Cloud (hosted vectors) | `WEAVIATE_URL`, `WEAVIATE_API_KEY` | Lab 2A, section 7 |
| A free LLM (Groq, Google AI Studio, Ollama, Azure OpenAI) | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` (Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`, `LLM_MODEL`) | Lab 2C |

### Option 1: inside the notebook (easiest)
**Lab 2A, section 7:** uncomment the two `os.environ[...]` lines at the top of the first cell of that section, paste your cluster URL and API key, run the cell.

**Lab 2C:** there is a ready-made cell right after the heading *A.0 Connect an LLM* (it starts with `# >>> PUT YOUR KEYS HERE`). Remove the leading `# ` from the three `os.environ` lines, paste your values, and run it **before** the next cell:

```python
import os
os.environ["LLM_API_KEY"]  = "PASTE-YOUR-KEY"
os.environ["LLM_BASE_URL"] = "https://api.groq.com/openai/v1"     # or the Google AI Studio / Ollama URL from the table below
os.environ["LLM_MODEL"]    = "PASTE-A-MODEL-NAME-FROM-YOUR-PROVIDER-CONSOLE"
```

Then run the next cell: it should print `LLM mode: LIVE ...`. If it still says offline demo mode, you set the values after the Helpers cell ran: run the cell above, then *Kernel > Restart & Run All*.

| Provider | `LLM_BASE_URL` | Key from |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | Groq console (free tier) |
| Google AI Studio | `https://generativelanguage.googleapis.com/v1beta/openai/` | Google AI Studio (free tier) |
| Ollama (local, no key) | `http://localhost:11434/v1` (set `LLM_MODEL` to the model you pulled; `LLM_API_KEY` can be any text) | none |

Model names change often: copy a current one from your provider's console.

### Option 2: environment variables before starting Jupyter
Mac / Linux:
```bash
export WEAVIATE_URL="https://YOUR-CLUSTER-ID.weaviate.cloud"
export WEAVIATE_API_KEY="PASTE-YOUR-KEY"
export LLM_API_KEY="PASTE-YOUR-KEY"
export LLM_BASE_URL="https://api.groq.com/openai/v1"
export LLM_MODEL="PASTE-A-MODEL-NAME"
jupyter lab
```
Windows PowerShell: use `$env:LLM_API_KEY="PASTE-YOUR-KEY"` (same pattern for the other variables), then `jupyter lab` in the same window.

### Keep your keys safe
- Never commit keys to GitHub and never share a notebook that has a key pasted in it (clear outputs and remove the cell first).
- Use synthetic or public data only with free hosted services.
- Delete or rotate your keys after the program.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate your virtual environment and run `pip install -r requirements.txt` again |
| Notebook cannot find `data/` or `llm_client` | Start Jupyter from inside the day folder (for example `day2/`) |
| Embedding model download fails | Tell the instructor; labs have fallbacks where possible |
| Weaviate Cloud section is skipped | Set `WEAVIATE_URL` and `WEAVIATE_API_KEY` first; it is optional, the Chroma sections cover the core ideas |

All data used in these labs is synthetic. Company names, documents and customers are fictional.
