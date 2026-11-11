#!/usr/bin/env python3
"""Prostuti auto question generator.

Flow per batch: generate -> rule checks -> dedupe -> independent solve check
(English + Bangla) -> save. Topics rotate; after the last topic it restarts
from the first and raises every topic's target by PER_TOPIC more questions.
Only Python standard library is used. API key comes from env GEMINI_API_KEY.
"""
import json, os, re, sys, time, hashlib, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(ROOT, "config.json"), encoding="utf-8"))
KEY = os.environ.get("GEMINI_API_KEY", "")
ORKEY = os.environ.get("OPENROUTER_API_KEY", "")
RUN_SECONDS = int(os.environ.get("RUN_MINUTES", CFG["run_minutes"])) * 60
MAX_CALLS = int(os.environ.get("MAX_CALLS", CFG["max_calls_per_run"]))
GAP = float(CFG["seconds_between_calls"])
OR_GAP = float(CFG.get("openrouter_gap", 4))
PER_TOPIC = int(CFG["per_topic"])
BATCH = int(CFG["batch_size"])
START = time.time()
calls = 0
dead_models = set()


class Stop(Exception):
    pass


def log(*a):
    print(*a, flush=True)


def load(path, default):
    try:
        return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))
    except Exception:
        return default


def save(path, data):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    tmp = full + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, full)


# ---------- Gemini ----------
def call_model(model, prompt, temperature):
    if model.startswith("or:"):
        return call_openrouter(model[3:], prompt, temperature)
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def call_openrouter(model, prompt, temperature):
    body = {"model": model, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + ORKEY})
    with urllib.request.urlopen(req, timeout=240) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"] or ""


def ask(models, prompt, temperature=0.7):
    """Try models in order. Raises Stop when everything is exhausted."""
    global calls
    for model in models:
        if model in dead_models:
            continue
        is_or = model.startswith("or:")
        if is_or and not ORKEY:
            dead_models.add(model); continue
        if not is_or and not KEY:
            dead_models.add(model); continue
        for attempt in range(3):
            if time.time() - START > RUN_SECONDS or calls >= MAX_CALLS:
                raise Stop("time or call budget finished")
            time.sleep(OR_GAP if is_or else GAP)
            calls += 1
            try:
                txt = call_model(model, prompt, temperature)
                return parse_json(txt)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "ignore")[:300]
                low = msg.lower()
                if e.code == 404:
                    log("model unavailable:", model); dead_models.add(model); break
                if e.code == 429:
                    if "perday" in low or "per day" in low or "per-day" in low or "limit: 0" in low or "daily" in low:
                        log("daily limit reached for", model); dead_models.add(model); break
                    log("rate limited, waiting 65s"); time.sleep(65); continue
                if e.code in (401, 402, 403):
                    log("not allowed (%d) for %s, skipping it" % (e.code, model)); dead_models.add(model); break
                if e.code >= 500:
                    time.sleep(20); continue
                log("HTTP", e.code, msg); dead_models.add(model); break
            except (ValueError, KeyError, IndexError, TypeError):
                log("bad answer format from", model, "- retrying"); continue
            except Exception as e:
                log("network problem:", e); time.sleep(20); continue
    raise Stop("no model available")


def parse_json(txt):
    txt = txt.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except ValueError:
        a, b = txt.find("["), txt.rfind("]")
        if a >= 0 and b > a:
            return json.loads(txt[a:b + 1])
        raise


# ---------- helpers ----------
BN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def norm(s):
    s = str(s).translate(BN).lower()
    return re.sub(r"[^\w]+", "", s)


def toks(s):
    s = str(s).translate(BN).lower()
    return set(re.findall(r"\w+", s))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def has_bangla(s):
    return bool(re.search(r"[\u0980-\u09FF]", s))


def clean_text(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def rule_check(q):
    try:
        for k in ("bn", "en", "ex", "xe"):
            q[k] = clean_text(q[k])
            if len(q[k]) < 8:
                return "short " + k
        o, oe = q["o"], q["oe"]
        if not (isinstance(o, list) and isinstance(oe, list) and len(o) == 4 and len(oe) == 4):
            return "need 4 options"
        q["o"] = [clean_text(x) for x in o]
        q["oe"] = [clean_text(x) for x in oe]
        if len(set(norm(x) for x in q["o"])) < 4 or len(set(norm(x) for x in q["oe"])) < 4:
            return "duplicate options"
        if any(not x for x in q["o"] + q["oe"]):
            return "empty option"
        a = q["a"]
        if not isinstance(a, int) or a < 0 or a > 3:
            return "bad answer index"
        if not has_bangla(q["bn"]) or not has_bangla(q["ex"]):
            return "bangla text missing"
        if re.search(r"https?:|www\.", json.dumps(q, ensure_ascii=False)):
            return "link inside"
        if re.search(r"(সব|উপরের সবগুলো|none of the above|all of the above)", " ".join(q["o"] + q["oe"]), re.I):
            return "all/none option"
        if q.get("d") not in ("e", "m", "h"):
            q["d"] = "m"
    except (KeyError, TypeError):
        return "missing field"
    return None


# ---------- prompts ----------
def gen_prompt(topic, existing, n, mix):
    stems = "\n".join("- " + x for x in existing[-60:]) or "(none yet)"
    return f"""You write high quality multiple-choice exam questions for Bangladeshi learners.
Topic: "{topic['title']}" (group: "{topic['group']}").
Write {n} NEW questions. Difficulty mix: {mix}.

Rules:
- Original questions only. Do not copy past exam papers word for word.
- Each question must have exactly ONE clearly correct option and 3 plausible wrong options.
- Only facts that are stable and widely agreed. Avoid anything that changes over time
  (current office holders, latest prices, "recent", "latest", "current", dated statistics).
  If you are not very sure of a fact, do not ask about it.
- No "all of the above" / "none of the above" options.
- Vary the sub-topics. Do not repeat or rephrase these existing questions:
{stems}
- Bangla fields must be natural, correct Bangla. English fields must be natural English.
- Explanation must show why the answer is right (for math show the working).

Return ONLY a JSON array. Each item exactly:
{{"d":"e|m|h","bn":"Bangla question","en":"English question",
"o":["4 Bangla options"],"oe":["same 4 options in English, same order"],
"a":0,"ex":"Bangla explanation","xe":"English explanation"}}
"a" is the 0-based index of the correct option (spread the correct position across 0..3)."""


def verify_prompt(items):
    rows = []
    for i, q in enumerate(items):
        rows.append({"id": i,
                     "english": {"question": q["en"], "options": q["oe"]},
                     "bangla": {"question": q["bn"], "options": q["o"]}})
    return ("You are a strict exam checker. For each item solve BOTH the English and the Bangla "
            "version yourself, carefully, before answering. Give the 0-based index of the single "
            "best option for each, or -1 if that version is ambiguous, has no correct option, has "
            "more than one correct option, or you are not sure. Return ONLY a JSON array like "
            "[{\"id\":0,\"en\":2,\"bn\":2},...].\n" + json.dumps(rows, ensure_ascii=False))


def solve_check(items):
    """Keep only questions where an independent solver agrees with the key in EN and BN."""
    res = ask(CFG["verifier_models"], verify_prompt(items), temperature=0.0)
    got = {}
    for r in res if isinstance(res, list) else []:
        try:
            got[int(r["id"])] = (int(r["en"]), int(r["bn"]))
        except (KeyError, ValueError, TypeError):
            pass
    return [q for i, q in enumerate(items) if got.get(i) == (q["a"], q["a"])]


# ---------- main ----------
def main():
    if not KEY:
        log("GEMINI_API_KEY is missing"); return 1
    log("OpenRouter key present:", bool(ORKEY))
    topics = load("topics.json", [])
    state = load("state.json", {"round": 1, "cursor": 0, "made": 0, "rejected": 0})
    # global exact-duplicate set
    allhash = set()
    qdir = os.path.join(ROOT, "questions")
    cache = {}
    for t in topics:
        data = load("questions/%s.json" % t["id"], [])
        cache[t["id"]] = data
        for q in data:
            allhash.add(norm(q["bn"]))
    log("topics:", len(topics), "round:", state["round"], "cursor:", state["cursor"])
    try:
        stuck = 0
        while True:
            if state["cursor"] >= len(topics):
                state["round"] += 1
                state["cursor"] = 0
                log("== all topics done, starting round", state["round"])
            t = topics[state["cursor"]]
            data = cache[t["id"]]
            target = PER_TOPIC * state["round"]
            if len(data) >= target:
                state["cursor"] += 1
                stuck = 0
                save("state.json", state)
                continue
            need = min(BATCH, target - len(data))
            mix = "about 30% easy, 50% medium, 20% hard"
            existing = [q["en"] for q in data]
            raw = ask(CFG["generator_models"], gen_prompt(t, existing, need, mix))
            if not isinstance(raw, list):
                raise Stop("generator returned non-list")
            fresh, rej = [], 0
            seen_toks = [toks(q["en"]) for q in data]
            for q in raw:
                if not isinstance(q, dict) or rule_check(q):
                    rej += 1; continue
                h = norm(q["bn"])
                tk = toks(q["en"])
                if h in allhash or any(jaccard(tk, x) >= 0.6 for x in seen_toks):
                    rej += 1; continue
                allhash.add(h); seen_toks.append(tk); fresh.append(q)
            good = solve_check(fresh) if fresh else []
            rej += len(fresh) - len(good)
            good = good[:max(0, target - len(data))]
            for q in good:
                data.append({"t": t["t"], "tp": t["title"], "d": q["d"], "bn": q["bn"], "en": q["en"],
                             "o": q["o"], "oe": q["oe"], "a": q["a"], "ex": q["ex"], "xe": q["xe"]})
            save("questions/%s.json" % t["id"], data)
            state["made"] += len(good)
            state["rejected"] += rej
            save("state.json", state)
            log("%s %s: +%d kept, %d rejected, total %d/%d" % (t["id"], t["title"], len(good), rej, len(data), target))
            stuck = stuck + 1 if not good else 0
            if stuck >= 6:  # this topic keeps failing, skip it for this round
                log("skipping topic for now:", t["title"])
                state["cursor"] += 1
                stuck = 0
                save("state.json", state)
    except Stop as e:
        log("stopping:", e)
    save("state.json", state)
    log("api calls used:", calls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
