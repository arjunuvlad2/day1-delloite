# Northwind Knowledge Assistant (milestone project)

Presenter: Nagarjuna Naidu

## What we build and why
A small, secure knowledge assistant for a fictional firm, Northwind. It answers HR, IT and finance questions from 48 company documents with sources, refuses when it does not know, respects who is allowed to read what, uses tools through an agent and an MCP server, is tracked and monitored, and is tested against attacks and unfair outcomes. The goal is to show that a complete, responsible AI solution needs surprisingly little code once the building blocks are in place. Every milestone notebook imports only from the small `nka` package.

## Folder map
| Path | What it is |
|---|---|
| `nka/` | The building blocks: config, data, index, tools, agent, mcp_server, mcp_client, security, evalkit, monitor, risk |
| `data/` | `corpus/` (48 documents) and `eval_queries.json` (golden questions) |
| `m1_tools.ipynb` ... `m6_multi_agent.ipynb` | The participant notebooks (open these), with `.py` script versions next to them |
| `extra_tools.py` | Tiny module that registers `lookup_policy` in the MCP server (used in M2) |
| `src/` | Notebook sources (trainer use) |
| `participant/`, `trainer/` | Built copies; trainer notebooks contain solutions and outputs |
| `build_labs.py` | Builds notebooks from `src/` (`python3 build_labs.py m1`) |
| `artifacts/` | Generated files: embedding cache, MLflow store, best config, logs |

## Milestones
| Milestone | Day | Time | Notebook |
|---|---|---|---|
| M1 RAG and an agent with tools | 4 | about 35 min | `m1_tools.ipynb` |
| M2 Tools over MCP | 4 | about 20 min | `m2_mcp.ipynb` |
| M3 Evaluate, track, monitor | 4 | about 30 min | `m3_monitor.ipynb` |
| M4 Secure and responsible AI | 5 | about 40 min | `m4_secure.ipynb` |
| M5 End-to-end demo and your scenario | 5 | remaining time | `m5_demo.ipynb` |
| M6 From one agent to a team of agents (before the case study) | 5 | about 15 min | `m6_multi_agent.ipynb` |

## Setup
```
pip install -r requirements.txt
```
Always start Jupyter (or Python) from this Project folder, so that `nka` and `data/` are found. Plain scripts: `PYTHONPATH=. python3 m1_tools.py`.

## Optional: connect a free LLM
No key is needed: without one, everything runs in **offline demo mode** (retrieval and quoted answers by simple rules, clearly labelled; it is not a real language model). For real generated answers, set these variables in the commented cell near the top of a notebook (or in your terminal before starting Jupyter):

| Provider | LLM_BASE_URL | LLM_MODEL (example) |
|---|---|---|
| Groq (free key) | `https://api.groq.com/openai/v1` | `openai/gpt-oss-20b` (check the list for your key) |
| Google AI Studio (free key) | `https://generativelanguage.googleapis.com/v1beta/openai/` | a current Gemini model from the AI Studio console |
| Ollama (local, no key) | `http://localhost:11434/v1` | the model you pulled |

Set `LLM_API_KEY` to your key (Ollama: any text or none). Azure OpenAI: set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` and `LLM_MODEL` to your deployment name. Keep keys out of files you share.

If you see a 404 or 'model not found', the helper prints the models your key can use: copy one of those chat models into `LLM_MODEL`.

## Tips
- After setting environment variables, **restart the kernel**; the settings are read once.
- Run the notebooks top to bottom. The first run downloads the small embedding model and caches it.
- Out of time? Offline mode is fine; the numbers and tables are the same.
- M2 starts a helper process; if it ever hangs, restart the kernel.
