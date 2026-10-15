"""Tiny LLM helper for the Day 2 labs.

Works with ANY OpenAI-compatible endpoint (OpenAI, Azure OpenAI, Groq, Google Gemini's
OpenAI-compatible endpoint, a local Ollama server, ...). Configure with environment variables:

    LLM_API_KEY    your key (not needed for a local Ollama server)
    LLM_BASE_URL   e.g. https://api.groq.com/openai/v1
                        https://generativelanguage.googleapis.com/v1beta/openai/
                        http://localhost:11434/v1          (Ollama)
    LLM_MODEL      model name offered by that provider

Azure OpenAI: set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION and
LLM_MODEL=<your deployment name>.

No configuration? Everything still runs in OFFLINE DEMO MODE: a tiny rule-based stand-in that is good
enough to see the pipeline work (retrieval + a quoted answer) but is NOT a real language model.
"""
import os, re, json

def _cfg():
    return dict(key=os.environ.get("LLM_API_KEY", ""), base=os.environ.get("LLM_BASE_URL", ""),
                model=os.environ.get("LLM_MODEL", ""), azure=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))

def llm_available() -> bool:
    c = _cfg()
    return bool(c["azure"] and os.environ.get("AZURE_OPENAI_API_KEY")) or bool(c["model"] and (c["key"] or c["base"].startswith("http://localhost")))

def llm_mode() -> str:
    c = _cfg()
    if not llm_available(): return "OFFLINE DEMO MODE (no LLM configured; see README to connect a free one)"
    return f"LIVE: {('azure:' + c['model']) if c['azure'] else c['base'] or 'api.openai.com'} / {c['model']}"

_STOP = set("a an the of to in on for and or is are was were be can i we you my our do does what how when which who it this that with at as by from about if not".split())
def _toks(s): return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP]

def _offline(user: str, system: str = None) -> str:
    """Deterministic stand-in. If the prompt carries numbered context blocks like [S1] ..., quote the best sentences."""
    blocks = re.findall(r"\[(S\d+)\]\s*(.*?)(?=\n\[S\d+\]|\Z)", user, flags=re.S)
    if blocks:
        q = re.search(r"(?:QUESTION|Question)\s*:?\s*(.*)", user)
        qt = set(_toks(q.group(1) if q else user[-300:]))
        best = []
        for sid, txt in blocks:
            txt = re.split(r"\n\s*(?:QUESTION|Question)\s*:", txt)[0]
            for sent in re.split(r"(?<=[.!?])\s+|\n+", txt):
                s = set(_toks(sent))
                if len(s) > 3: best.append((len(qt & s) / (1 + 0.1 * len(s)), sid, sent.strip()))
        best.sort(reverse=True)
        top = [b for b in best[:2] if b[0] > 0]
        if not top: return "I could not find this in the provided documents."
        return "Based on the documents: " + " ".join(f"{t} [{sid}]" for _, sid, t in top) + "  (offline demo mode: quoted, not generated)"
    return "[offline demo mode] No LLM is connected, so this step cannot produce a real answer. See the README to connect a free one."

_TOOL_ERR = ("tool choice is none", "called a tool", "tool_use_failed", "failed_generation")
def _tool_call_error(e) -> bool:
    return any(t in str(e).lower() for t in _TOOL_ERR)

def _salvage(e):
    """If the provider returns the model's rejected output (failed_generation) and it is valid JSON, use it."""
    try:
        body = getattr(e, "body", None) or {}
        txt = (body.get("error", body) or {}).get("failed_generation", "") if isinstance(body, dict) else ""
        i, j = txt.find("{"), txt.rfind("}")
        if i >= 0 and j > i:
            obj = json.loads(txt[i:j + 1])
            if isinstance(obj, dict): return json.dumps(obj)
    except Exception:
        pass
    return None

def _no_tools(msgs, strong=False):
    note = (" IMPORTANT: no tools or functions exist in this chat. Never call a tool, function or browser; "
            "write your answer directly as plain text" + (" (or the JSON object requested)." if True else ".")) + (" Do not output tool-call syntax." if strong else "")
    out = [dict(m) for m in msgs]
    if out and out[0]["role"] == "system":
        if note not in out[0]["content"]: out[0]["content"] += note
    else:
        out.insert(0, {"role": "system", "content": note.strip()})
    return out

def call_llm(user: str, system: str = None, temperature: float = 0.2, max_tokens: int = 500,
             messages: list = None, json_mode: bool = False) -> str:
    """Return the assistant's text. Falls back to offline demo mode when no LLM is configured or the call fails."""
    if not llm_available(): return _offline(user, system)
    try:
        from openai import OpenAI, AzureOpenAI
        c = _cfg()
        if c["azure"]:
            client = AzureOpenAI(azure_endpoint=c["azure"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                 api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01"))
        else:
            client = OpenAI(api_key=c["key"] or "not-needed", base_url=c["base"] or None)
        msgs = messages or ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = dict(model=c["model"], messages=msgs, temperature=temperature, max_tokens=max_tokens)
        oss = "gpt-oss" in c["model"].lower()
        if oss:      # reasoning models: keep thinking short, leave room for the answer, and tell them no tools exist
            kw["max_tokens"] = max(max_tokens, 1500); kw["extra_body"] = {"reasoning_effort": "low"}
            kw["messages"] = _no_tools(msgs)
        def _ask(kwargs):
            out = (client.chat.completions.create(**kwargs).choices[0].message.content or "").strip()
            if not out: raise ValueError("the model returned an empty answer (try a larger max_tokens or another model)")
            return out
        def _try(kwargs):
            last = None
            for attempt in range(3):             # gpt-oss sometimes tries to call a tool unasked: retry, then salvage
                try: return _ask(kwargs)
                except Exception as e:
                    last = e
                    if not _tool_call_error(e): raise
                    saved = _salvage(e) if json_mode else None
                    if saved: return saved
                    kwargs = {**kwargs, "temperature": 0.0, "messages": _no_tools(kwargs["messages"], strong=True)}
            raise last
        if json_mode:
            try: return _try({**kw, "response_format": {"type": "json_object"}})
            except Exception as e:
                if _tool_call_error(e): raise
                return _try(kw)                  # some models do not support JSON mode: retry as a normal call
        return _try(kw)
    except Exception as e:                       # never break a live demo: say what happened and fall back
        print(f"[llm_client] call failed ({type(e).__name__}: {str(e)[:100]}); using offline demo mode")
        if "NotFound" in type(e).__name__ or "404" in str(e) or "does not exist" in str(e):
            try:        # wrong model name: show the models this key can actually use
                ids = sorted(m.id for m in client.models.list().data)
                print("[llm_client] models available to your key (set LLM_MODEL to one of these chat models):\n  " + "\n  ".join(ids[:25]))
            except Exception:
                print("[llm_client] could not list models: check LLM_API_KEY and LLM_BASE_URL.")
        if _tool_call_error(e):
            print("[llm_client] this model keeps trying to call tools on this provider; set LLM_MODEL to another chat model (for Groq, try openai/gpt-oss-120b or qwen/qwen3.8-27b from your model list).")
        return _offline(user, system)
