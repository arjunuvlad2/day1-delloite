"""A small tool-using agent: planner (live LLM or offline rules) + guarded loop + final answer."""
import inspect, json, re
from . import llm_client
from .config import MIN_SIM
from .tools import TOOLS, SCHEMA

REFUSAL = "I could not find this in the provided documents."
NUM = r"\d[\d,]*\.?\d*"
POLICY_ID = r"\bPOL-[A-Z]+-\d+(?:-[A-Z])?\b"


def _num(s): return float(s.replace(",", ""))


def _finish(thought): return dict(thought=thought, action="finish", input="")


def plan_offline(question, scratch, tools=TOOLS):
    """OFFLINE rule-based stand-in for the LLM planner (it is not a language model)."""
    q, done = question.lower(), [s["action"] for s in scratch]
    m = re.search(rf"({NUM})\s*(?:%|percent)\s*of\s*({NUM})", q)
    pid = re.search(POLICY_ID, question, re.I)
    if m and not scratch:
        return dict(thought="Pure arithmetic, no documents needed.", action="calculator",
                    input=f"{_num(m.group(1)):g} / 100 * {_num(m.group(2)):g}")
    if not scratch and not re.search(r"[a-z]{4,}", re.sub(r"what|is|of|how|much|percent|times|plus|minus", "", q)) and re.search(r"\d\s*[-+*/x]\s*\d", q):
        return dict(thought="Simple arithmetic.", action="calculator", input=re.sub(r"[^0-9+\-*/(). ]", "", q.replace("x", "*")))
    if pid and "lookup_policy" in tools:
        if not scratch: return dict(thought=f"The question names {pid.group().upper()}; look it up.", action="lookup_policy", input=pid.group().upper())
        return _finish("I have the policy record.")
    if "booking" in q and "cap" in q:                                   # documents -> sql -> calculator
        if "search_docs" not in done:
            return dict(thought="I need the policy hotel cap first.", action="search_docs", input="hotel cap per night London")
        cap = re.search(r"London\s+(?:GBP|EUR|USD)\s+(\d+)", scratch[0]["observation"])
        if not cap: return _finish("The cap was not in the search results.")
        c = cap.group(1)
        if "query_sql" not in done:
            return dict(thought=f"The cap is {c}. Now find London bookings above it.", action="query_sql",
                        input=f"SELECT traveller, nights, price_per_night FROM bookings WHERE city='London' AND currency='GBP' AND price_per_night > {c} ORDER BY traveller")
        if "calculator" not in done:
            rows = json.loads(scratch[1]["observation"])
            expr = " + ".join(f"{r['nights']} * ({r['price_per_night']:g} - {c})" for r in rows) or "0"
            return dict(thought="Total overage = sum of nights * (price - cap).", action="calculator", input=expr)
        return _finish("I have the total overage.")
    if not scratch:
        return dict(thought="I need facts from the documents first.", action="search_docs", input=question)
    return _finish("I have what the documents say." if done[-1] == "search_docs" else "I have the result.")


def plan_llm(question, scratch, tools=TOOLS):
    """Ask the live LLM for ONE next step as JSON; anything off the allow-list becomes 'finish'."""
    tl = "\n".join(f"- {n}: {t['description']}" for n, t in tools.items()) + "\n- finish: give the final answer"
    pad = "\n".join(f"Step {i}: {s['action']}({s['input']}) -> {s['observation'][:400]}" for i, s in enumerate(scratch, 1)) or "(empty)"
    prompt = (f"Tools:\n{tl}\n\nTable: {SCHEMA}\n\nScratchpad:\n{pad}\n\nQUESTION: {question}\n"
              'Reply with JSON only: {"thought": "...", "action": "<tool name or finish>", "input": "..."}')
    try:
        out = json.loads(llm_client.call_llm(prompt, system="You are a careful planner. Pick ONE next action. The tools below are NOT callable functions: do not use function calling; reply with the JSON object only.", json_mode=True))
        if out.get("action") in list(tools) + ["finish"]:
            return dict(thought=out.get("thought", ""), action=out["action"], input=str(out.get("input", "")))
        return _finish(f"Planner chose a disallowed action: {out.get('action')!r}")
    except Exception as e:      # live model failed or returned something unusable: keep going with the offline rules
        print(f"[agent] live planner failed ({type(e).__name__}); using the offline rules for this step")
        return plan_offline(question, scratch, tools)


def plan(question, scratchpad, tools=TOOLS) -> dict:
    """Next step as dict(thought, action, input). Live LLM when configured, else the offline rules."""
    return (plan_llm if llm_client.llm_available() else plan_offline)(question, scratchpad, tools)


def _call(tool, text, role):
    fn = tool["fn"]
    return fn(text, role=role) if "role" in inspect.signature(fn).parameters else fn(text)


def final_answer(question, scratch):
    ok = {s["action"]: s for s in scratch if not s["observation"].startswith("ERROR")}
    if llm_client.llm_available():
        pad = "\n".join(f"{s['action']}({s['input']}) -> {s['observation'][:600]}" for s in scratch)
        out = llm_client.call_llm(f"{pad}\n\nQUESTION: {question}",
                                  system="Answer only from the scratchpad; if it lacks the answer say 'I could not find this'.")
        if out and not out.startswith("[offline demo mode]"): return out     # live answer; otherwise use the offline rules below
    if "query_sql" in ok and "calculator" in ok and "search_docs" in ok:
        cap = re.search(r"London\s+GBP\s+(\d+)", ok["search_docs"]["observation"]).group(1)
        rows = json.loads(ok["query_sql"]["observation"])
        lines = "\n".join(f"  - {r['traveller']}: {r['nights']} nights at GBP {r['price_per_night']:g}" for r in rows)
        return f"{len(rows)} London bookings exceed the GBP {cap} cap:\n{lines}\nTotal over the cap: GBP {float(ok['calculator']['observation']):g}."
    if "calculator" in ok: return f"The result is {float(ok['calculator']['observation']):,g}."
    if "lookup_policy" in ok: return ok["lookup_policy"]["observation"]
    if "search_docs" in ok:
        obs = ok["search_docs"]["observation"]
        found = re.findall(r"(?ms)^\[(.+?)\] \(sim ([\d.]+)\) (.*?)(?=^\[.+?\] \(sim |\Z)", obs)
        if not found or float(found[0][1]) < MIN_SIM: return REFUSAL
        blocks = "\n".join(f"[S{i}] {t.strip()}" for i, (_, _, t) in enumerate(found, 1))
        srcs = "; ".join(dict.fromkeys(t for t, _, _ in found))
        return llm_client.call_llm(f"{blocks}\n\nQUESTION: {question}") + f"\n  Sources: {srcs}"
    return REFUSAL


def run_agent(question, max_steps=5, role="employee", verbose=True, tools=TOOLS) -> dict:
    """Plan -> act -> observe loop. Guardrails: step limit, tool allow-list, tool errors caught."""
    scratch, say = [], (print if verbose else (lambda *a, **k: None))
    say(f"\nTASK: {question}")
    for step in range(1, max_steps + 1):
        p = plan(question, scratch, tools)
        say(f"\n[{step}] Thought: {p['thought']}\n    Action: {p['action']}({p['input']})")
        if p["action"] == "finish": break
        if p["action"] not in tools: say("    blocked: tool not on allow-list"); break
        try: obs = str(_call(tools[p["action"]], p["input"], role))
        except Exception as e: obs = f"ERROR: {e}"
        say("    Observation:", obs[:160].replace("\n", " ") + ("..." if len(obs) > 160 else ""))
        scratch.append(dict(p, observation=obs))
    answer = final_answer(question, scratch)
    say("FINAL ANSWER:", answer)
    return dict(answer=answer, steps=scratch, tool_calls=len(scratch), mode="live" if llm_client.llm_available() else "offline")
