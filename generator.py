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
GEM_EXTRA, OR_EXTRA = [], []
used_models = set()


def http_json(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def discover():
    """Ask Gemini and OpenRouter which models exist today, so renamed or retired
    models never stop the work."""
    global GEM_EXTRA, OR_EXTRA
    try:
        data = http_json("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                         {"x-goog-api-key": KEY})
        names = []
        for m in data.get("models", []):
            n = m.get("name", "").replace("models/", "")
            if not n.startswith("gemini"):
                continue
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            if any(b in n for b in ("image", "tts", "live", "embed", "audio", "robotics", "banana", "imagen", "veo", "nano", "omni",
                                    "computer", "vision", "learnlm", "thinking-exp")):
                continue
            names.append(n)

        def rank(n):
            fam = 0 if ("flash" in n and "lite" not in n) else (1 if "pro" in n else 2)
            nums = re.findall(r"\d+(?:\.\d+)?", n)
            ver = 99.0 if "latest" in n else (float(nums[0]) if nums else 0.0)
            pre = 1 if ("preview" in n or "exp" in n) else 0
            return (fam, pre, -ver, n)
        GEM_EXTRA = sorted(set(names), key=rank)
        log("Gemini models found:", ", ".join(GEM_EXTRA[:8]) or "none")
    except Exception as e:
        log("could not list Gemini models:", e)
    if ORKEY:
        try:
            data = http_json("https://openrouter.ai/api/v1/models", {})
            pref = ("gpt-oss-120b", "qwen3", "nemotron", "deepseek", "llama-3.3", "gemma", "glm", "kimi", "mistral")
            rows = []
            for m in data.get("data", []):
                i = m.get("id", "")
                if not i.endswith(":free"):
                    continue
                out = (m.get("architecture") or {}).get("output_modalities") or ["text"]
                if "text" not in out:
                    continue
                if any(b in i for b in ("-vl", "vision", "image", "audio", "embed", "guard", "safety", "lyria")):
                    continue
                idx = next((k for k, x in enumerate(pref) if x in i), 99)
                rows.append((idx, -(m.get("context_length") or 0), i))
            OR_EXTRA = ["or:" + r[2] for r in sorted(rows)] + ["or:openrouter/free"]
            log("OpenRouter free models found:", ", ".join(OR_EXTRA[:6]) or "none")
        except Exception as e:
            log("could not list OpenRouter models:", e)


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


cooldown = {}   # model -> time when it may be used again
fmt_fail = {}


def retry_seconds(msg):
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.I) or re.search(r'retryDelay"?\s*:\s*"(\d+(?:\.\d+)?)s', msg)
    return float(m.group(1)) if m else 60.0


last_ok = [None]


def ask(models, prompt, temperature=0.7, role="gen", avoid=None):
    """Use the best model that is ready. A model that is rate limited or busy is
    put to rest and the next model is used at once. Raises Stop when nothing is left."""
    global calls
    if role == "verify":
        cands = [m for m in models if m.startswith("or:")] + OR_EXTRA + \
                [m for m in models if not m.startswith("or:")] + GEM_EXTRA
    else:
        cands = [m for m in models if not m.startswith("or:")] + GEM_EXTRA
    seen = set()
    cands = [m for m in cands if not (m in seen or seen.add(m)) and m != avoid]
    while True:
        now = time.time()
        if now - START > RUN_SECONDS or calls >= MAX_CALLS:
            raise Stop("time or call budget finished")
        live = []
        for m in cands:
            if m in dead_models:
                continue
            if (m.startswith("or:") and not ORKEY) or (not m.startswith("or:") and not KEY):
                dead_models.add(m); continue
            live.append(m)
        if not live:
            raise Stop("no model available (tried %d models)" % len(cands))
        ready = [m for m in live if cooldown.get(m, 0) <= now]
        if not ready:
            wait = min(cooldown[m] for m in live) - now
            if now + wait - START > RUN_SECONDS:
                raise Stop("all models resting until the time budget ends")
            log("all models resting, waiting %ds" % max(5, wait))
            time.sleep(min(max(wait, 5), 300))
            continue
        model = ready[0]
        is_or = model.startswith("or:")
        time.sleep(OR_GAP if is_or else GAP)
        calls += 1
        try:
            txt = call_model(model, prompt, temperature)
            res = parse_json(txt)
            fmt_fail[model] = 0
            last_ok[0] = model
            if model not in used_models:
                used_models.add(model); log("working model:", model)
            return res
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:3000]
            low = msg.lower()
            if e.code == 404:
                log("model unavailable:", model); dead_models.add(model)
            elif e.code == 429:
                if "perday" in low or "per day" in low or "per-day" in low or "limit: 0" in low or "daily" in low:
                    log("daily limit reached for", model); dead_models.add(model)
                    if is_or:  # all free OpenRouter models share one daily limit
                        for m2 in cands:
                            if m2.startswith("or:"):
                                dead_models.add(m2)
                else:
                    wait = min(retry_seconds(msg), 300) + 2
                    log("rate limited: %s rests %ds, trying another model" % (model, wait))
                    cooldown[model] = time.time() + wait
            elif e.code in (401, 402, 403):
                log("not allowed (%d) for %s, skipping it" % (e.code, model)); dead_models.add(model)
            elif e.code >= 500:
                log("server error %d from %s, it rests 45s" % (e.code, model))
                cooldown[model] = time.time() + 45
            else:
                log("HTTP", e.code, msg[:200]); dead_models.add(model)
        except (ValueError, KeyError, IndexError, TypeError):
            fmt_fail[model] = fmt_fail.get(model, 0) + 1
            log("bad answer format from", model)
            if fmt_fail[model] >= 2:
                fmt_fail[model] = 0
                cooldown[model] = time.time() + 600
        except Exception as e:
            log("network problem with %s: %s - it rests 90s" % (model, e))
            cooldown[model] = time.time() + 90


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


def latin_words(x):
    return len(re.findall(r"[A-Za-z]{2,}", x))


def lang_problem(q, ttype=None):
    """English fields must be pure English (English options in the Bangla options are allowed)."""
    if has_bangla(q["en"]) or has_bangla(q["xe"]) or any(has_bangla(x) for x in q["oe"]):
        return "bangla inside english fields"
    return None


def rule_check(q, ttype=None):
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
        if any(re.match(r"\s*(সবগুলো|সবকটি|সবক’টি|উপরের (সব|কোনো)|কোনোটিই নয়|কোনটিই নয়|(all|none) of the (above|these))", x, re.I)
               for x in q["o"] + q["oe"]):
            return "all/none option"
        lang = lang_problem(q, ttype)
        if lang:
            return lang
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
- Language rules: the English fields (en, oe, xe) must contain NO Bangla letters.
  In the Bangla options (o), when an answer is a name, place, period, title, term or phrase
  that has an English form, write the Bangla first and the English in brackets right after it,
  e.g. "এলিজাবেথীয় যুগ (The Elizabethan Age)", "সালোকসংশ্লেষণ (Photosynthesis)".
  Universal acronyms and technical terms (HTTP, FTP, DNA, CPU) stay as they are. Plain numbers,
  years and calculated values get no brackets. For topics about the English language itself,
  options may stay in English. The Bangla question (bn) may keep a key English term in brackets.
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


def solve_check(items, gen_model=None):
    """Keep only questions where an independent solver agrees with the key in EN and BN."""
    res = ask(CFG["verifier_models"], verify_prompt(items), temperature=0.0, role="verify", avoid=gen_model)
    got = {}
    for r in res if isinstance(res, list) else []:
        try:
            got[int(r["id"])] = (int(r["en"]), int(r["bn"]))
        except (KeyError, ValueError, TypeError):
            pass
    return [q for i, q in enumerate(items) if got.get(i) == (q["a"], q["a"])]



CLEAN_VERSION = 2


def clean_saved(topics, cache, state):
    """One-time pass: drop saved questions that break the language rules (they get regenerated)."""
    if state.get("clean", 0) >= CLEAN_VERSION:
        return
    removed = 0
    for t in topics:
        data = cache[t["id"]]
        keep = []
        for q in data:
            try:
                bad = lang_problem(q, t["t"])
            except (KeyError, TypeError):
                bad = "broken"
            if bad:
                removed += 1
            else:
                keep.append(q)
        if len(keep) != len(data):
            cache[t["id"]] = keep
            save("questions/%s.json" % t["id"], keep)
    state["clean"] = CLEAN_VERSION
    save("state.json", state)
    log("cleanup: removed %d saved questions with mixed languages" % removed)


# ---------- main ----------
def main():
    if not KEY:
        log("GEMINI_API_KEY is missing"); return 1
    log("OpenRouter key present:", bool(ORKEY))
    discover()
    topics = load("topics.json", [])
    state = load("state.json", {"round": 1, "cursor": 0, "made": 0, "rejected": 0})
    cache = {}
    for t in topics:
        cache[t["id"]] = load("questions/%s.json" % t["id"], [])
    clean_saved(topics, cache, state)
    # global exact-duplicate set
    allhash = set()
    for t in topics:
        for q in cache[t["id"]]:
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
            raw = ask(CFG["generator_models"], gen_prompt(t, existing, need, mix), role="gen")
            gen_model = last_ok[0]
            if not isinstance(raw, list):
                raise Stop("generator returned non-list")
            fresh, rej = [], 0
            r_rule = r_dup = 0
            seen_toks = [toks(q["en"]) for q in data]
            for q in raw:
                if not isinstance(q, dict) or rule_check(q, t["t"]):
                    rej += 1; r_rule += 1; continue
                h = norm(q["bn"])
                tk = toks(q["en"])
                if h in allhash or any(jaccard(tk, x) >= 0.6 for x in seen_toks):
                    rej += 1; r_dup += 1; continue
                allhash.add(h); seen_toks.append(tk); fresh.append(q)
            good = solve_check(fresh, gen_model) if fresh else []
            r_ver = len(fresh) - len(good)
            rej += r_ver
            good = good[:max(0, target - len(data))]
            for q in good:
                data.append({"t": t["t"], "tp": t["title"], "d": q["d"], "bn": q["bn"], "en": q["en"],
                             "o": q["o"], "oe": q["oe"], "a": q["a"], "ex": q["ex"], "xe": q["xe"]})
            save("questions/%s.json" % t["id"], data)
            state["made"] += len(good)
            state["rejected"] += rej
            save("state.json", state)
            log("%s %s: +%d kept, %d rejected (format %d, duplicate %d, solver disagreed %d), total %d/%d" % (
                t["id"], t["title"], len(good), rej, r_rule, r_dup, r_ver, len(data), target))
            stuck = stuck + 1 if not good else 0
            if stuck >= 6:  # this topic keeps failing, skip it for this round
                log("skipping topic for now:", t["title"])
                state["cursor"] += 1
                stuck = 0
                save("state.json", state)
    except Stop as e:
        log("stopping:", e)
    save("state.json", state)
    log("api calls used:", calls, "| minutes:", round((time.time() - START) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
