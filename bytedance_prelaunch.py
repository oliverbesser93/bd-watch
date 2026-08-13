#!/usr/bin/env python3
"""
bytedance_prelaunch.py -- Fruehwarnsystem fuer kommende ByteDance-Modelle.

Sucht nach Signalen VOR dem offiziellen Launch:

  1. framework  PRs/Issues in vLLM, SGLang, transformers, llama.cpp, die eine
                ByteDance-Architektur einbauen  (Vorlauf: 1-4 Wochen)
  2. hf_empty   HF-Repos mit Config, aber noch ohne Gewichte oder gated
                -> Upload laeuft gerade  (Vorlauf: Stunden-Tage)
  3. arxiv      Neue Paper des Seed-Teams  (Vorlauf: Wochen)
  4. pages      Neue Modellnamen auf Vendor-Seiten  (Vorlauf: Tage-Wochen)

Treffer aus 1. werden bewertet (0-10); nur Treffer ab MIN_SCORE loesen eine
Benachrichtigung aus. Der Rest wandert stumm in den Zustand.

Nutzung:
  python3 bytedance_prelaunch.py --init            # Ist-Stand merken
  python3 bytedance_prelaunch.py                   # pruefen und melden
  python3 bytedance_prelaunch.py --min-score 3     # Schwelle senken
  python3 bytedance_prelaunch.py --only pages,hf_empty
  python3 bytedance_prelaunch.py --dry-run         # nichts speichern/senden

Alle Quellen sind oeffentlich; kein Login, keine Umgehung von Schutzmassnahmen.
Nicht haeufiger als stuendlich laufen lassen.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from html import unescape

# --- Konfiguration ------------------------------------------------------

STATE_FILE = os.environ.get(
    "BD_STATE_FILE", os.path.expanduser("~/.bytedance_prelaunch_state.json")
)
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
MIN_SCORE = int(os.environ.get("BD_MIN_SCORE", "5"))

HF_AUTHORS = ["ByteDance", "ByteDance-Seed", "bytedance-research"]

FRAMEWORK_REPOS = [
    "vllm-project/vllm",
    "sgl-project/sglang",
    "huggingface/transformers",
    "ggml-org/llama.cpp",
    "hiyouga/LLaMA-Factory",
]
FRAMEWORK_TERMS = ["bytedance", "seed-oss", "doubao", "seedream", "ui-tars"]

WATCH_PAGES = [
    "https://seed.bytedance.com/en/blog",
    "https://seed.bytedance.com/en/model",
]

MODEL_RE = re.compile(
    r"\b(Seed(?:ance|ream|Edit|Realtime|Stream|LM|VL)?|Doubao|UI-TARS|BAGEL|Goku)"
    r"[\s\-_]?(?:v)?(\d+(?:\.\d+)?)"
    r"((?:[\s\-](?:Pro|Lite|Preview|Flash|Turbo|Max|Mini|Thinking|Omni|Base))*)",
    re.IGNORECASE,
)

UA = "bytedance-prelaunch/1.1 (personal research monitor)"
TIMEOUT = 30

# --- Relevanzbewertung --------------------------------------------------

# (Regex, Punkte, Begruendung)
POSITIVE = [
    (r"\badd(?:ing|s)?\s+(?:support\s+for|model|new\s+model)", 4, "add support"),
    (r"\bsupport\s+for\s+\w", 3, "support for"),
    (r"\b(?:new|initial)\s+model\b", 4, "new model"),
    (r"\bmodel\s+support\b", 4, "model support"),
    (r"\bimplement(?:s|ing)?\b", 2, "implement"),
    (r"\barchitecture\b", 2, "architecture"),
    (r"\bForCausalLM\b|\bconfig(?:uration)?_class\b", 3, "arch-code"),
    (r"\bregist(?:er|ry)\b", 1, "registry"),
    (r"\bMoE\b|\bmixture[- ]of[- ]experts\b", 1, "MoE"),
    (r"\bconvert(?:er|ing)?\s+(?:script|weights)", 2, "converter"),
]
NEGATIVE = [
    (r"\[usage\]|\[bug\]|\[help\]|\[question\]", -5, "usage/bug-Tag"),
    (r"\b(?:bug|crash|traceback|segfault|regression)\b", -3, "Fehlerreport"),
    (r"\b(?:oom|out of memory|slow|latency|throughput)\b", -3, "Performance"),
    (r"\b(?:typo|readme|changelog|comment)\b", -4, "Kosmetik"),
    (r"\bhow (?:to|do i)\b|\bquestion\b", -3, "Frage"),
    (r"\bfix(?:es|ed)?\b", -2, "Fix"),
    (r"\bbenchmark|\beval(?:uation)?\b", -1, "Benchmark"),
]
GOOD_LABELS = {"new-model", "new model", "model", "feature", "enhancement"}


def score_issue(item, known_names):
    """0-10 Punkte plus Liste der Gruende."""
    text = f"{item.get('title','')} {(item.get('body') or '')[:2000]}"
    low = text.lower()
    score, why = 0, []

    for pat, pts, label in POSITIVE + NEGATIVE:
        if re.search(pat, low, re.I):
            score += pts
            why.append(f"{label}{pts:+d}")

    if "pull_request" in item:
        score += 2
        why.append("PR+2")

    labels = {l["name"].lower() for l in item.get("labels", [])}
    if labels & GOOD_LABELS:
        score += 3
        why.append("Label+3")

    # Der eigentliche Jackpot: ein Modellname, den wir noch nie gesehen haben
    found = {normalize_model(m) for m in MODEL_RE.finditer(text)}
    new_names = {n for n in found if n.lower() not in known_names}
    if new_names:
        score += 5
        why.append("neuer Name: " + ", ".join(sorted(new_names)))
    elif found:
        score += 1
        why.append("bekanntes Modell")

    return max(0, min(10, score)), why, found


# --- Helfer -------------------------------------------------------------


def fetch(url, headers=None, as_json=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read()
    return json.loads(raw) if as_json else raw.decode("utf-8", "ignore")


def gh_headers():
    return {"Authorization": f"Bearer {GH_TOKEN}"} if GH_TOKEN else {}


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"seen": [], "names": []}


def save_state(state):
    os.makedirs(os.path.dirname(os.path.abspath(STATE_FILE)) or ".", exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def notify(title, body, url="", priority="default"):
    print(f"\n### {title}\n{body}\n{url}", flush=True)
    if not NTFY_TOPIC:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=body.encode("utf-8"),
            headers={
                "User-Agent": UA,
                "Title": title.encode("ascii", "ignore").decode(),
                "Click": url or "https://seed.bytedance.com/en/blog",
                "Priority": priority,
                "Tags": "satellite",
            },
        )
        urllib.request.urlopen(req, timeout=TIMEOUT).read()
    except Exception as e:
        print(f"[warn] ntfy: {e}", file=sys.stderr)


def strip_html(html):
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html)))


def normalize_model(m):
    fam, ver, suffix = m.group(1), m.group(2), m.group(3)
    suffix = re.sub(r"[\s\-]+", "-", suffix.strip()).strip("-")
    return f"{fam.title()} {ver}{('-' + suffix) if suffix else ''}"


# --- Signal 1: Framework-Support ---------------------------------------


def check_framework(seen, state):
    known = set(state.get("names", []))
    hits, skipped = [], 0
    for repo in FRAMEWORK_REPOS:
        for term in FRAMEWORK_TERMS:
            q = urllib.parse.quote(f"repo:{repo} {term} in:title,body sort:created")
            url = (
                f"https://api.github.com/search/issues?q={q}"
                "&per_page=10&sort=created&order=desc"
            )
            try:
                data = fetch(url, gh_headers(), as_json=True)
            except Exception as e:
                print(f"[warn] GH-Suche {repo}/{term}: {e}", file=sys.stderr)
                continue
            for it in data.get("items", []):
                key = f"gh:{it['id']}"
                if key in seen:
                    continue
                score, why, found = score_issue(it, known)
                state.setdefault("names", [])
                for n in found:
                    if n.lower() not in known:
                        known.add(n.lower())
                        state["names"].append(n.lower())
                if score < MIN_SCORE:
                    skipped += 1
                    seen.add(key)  # gesehen, aber nicht melden
                    continue
                kind = "PR" if "pull_request" in it else "Issue"
                hits.append(
                    (
                        key,
                        f"[{score}/10] {repo}: {it['title'][:80]}",
                        f"{kind} von {it['user']['login']}, {it['created_at'][:10]}\n"
                        f"Signale: {', '.join(why) or '-'}",
                        it["html_url"],
                        score,
                    )
                )
            time.sleep(3)  # Search-API: 10 req/min ohne Token
    if skipped:
        print(f"[info] framework: {skipped} Treffer unter Schwelle {MIN_SCORE} ignoriert")
    return hits


# --- Signal 2: HF-Repo ohne Gewichte -----------------------------------


def check_hf_empty(seen, state):
    hits = []
    for author in HF_AUTHORS:
        q = urllib.parse.urlencode(
            {"author": author, "sort": "createdAt", "direction": -1,
             "limit": 30, "full": "true"}
        )
        try:
            models = fetch(f"https://huggingface.co/api/models?{q}", as_json=True)
        except Exception as e:
            print(f"[warn] HF {author}: {e}", file=sys.stderr)
            continue
        for m in models:
            files = [s.get("rfilename", "") for s in m.get("siblings", [])]
            has_weights = any(
                f.endswith((".safetensors", ".bin", ".gguf", ".pt")) for f in files
            )
            gated = bool(m.get("gated"))
            if has_weights and not gated:
                continue
            key = f"hf:{m['id']}:{'gated' if gated else 'noweights'}"
            if key in seen:
                continue
            reason = "gated (Zugang beschraenkt)" if gated else "noch keine Gewichte"
            hits.append(
                (
                    key,
                    f"HF-Repo im Aufbau: {m['id']}",
                    f"{reason} | {len(files)} Dateien | angelegt {m.get('createdAt','')[:10]}",
                    f"https://huggingface.co/{m['id']}",
                    8,
                )
            )
    return hits


# --- Signal 3: arXiv ----------------------------------------------------


def check_arxiv(seen, state):
    q = urllib.parse.urlencode(
        {
            "search_query": 'all:"ByteDance Seed"',
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": 20,
        }
    )
    try:
        xml = fetch(f"http://export.arxiv.org/api/query?{q}")
    except Exception as e:
        print(f"[warn] arXiv: {e}", file=sys.stderr)
        return []
    hits = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        link = re.search(r"<id>(.*?)</id>", entry)
        title = re.search(r"<title>(.*?)</title>", entry, re.S)
        if not link:
            continue
        key = f"arxiv:{link.group(1)}"
        if key in seen:
            continue
        t = re.sub(r"\s+", " ", title.group(1)).strip() if title else "?"
        hits.append((key, "Neues Seed-Paper", t, link.group(1), 6))
    return hits


# --- Signal 4: Modellnamen auf Vendor-Seiten ---------------------------


def check_pages(seen, state):
    known = set(state.get("names", []))
    hits = []
    for url in WATCH_PAGES:
        try:
            text = strip_html(fetch(url))
        except Exception as e:
            print(f"[warn] Seite {url}: {e}", file=sys.stderr)
            continue
        soon = "coming soon" in text.lower()
        for m in MODEL_RE.finditer(text):
            name = normalize_model(m)
            if name.lower() in known:
                continue
            known.add(name.lower())
            state.setdefault("names", []).append(name.lower())
            ctx = text[max(0, m.start() - 90): m.end() + 90].strip()
            hits.append(
                (
                    f"name:{name.lower()}",
                    f"Unbekannter Modellname: {name}",
                    ("[coming soon] " if soon else "") + "..." + ctx + "...",
                    url,
                    10,
                )
            )
    return hits


# --- Hauptlauf ----------------------------------------------------------

CHECKS = {
    "framework": check_framework,
    "hf_empty": check_hf_empty,
    "arxiv": check_arxiv,
    "pages": check_pages,
}


def run(init=False, only=None, dry=False):
    state = load_state()
    seen = set(state.get("seen", []))
    names = [n.strip() for n in only.split(",")] if only else list(CHECKS)

    all_hits = []
    for name in names:
        fn = CHECKS.get(name)
        if not fn:
            print(f"[warn] unbekannter Check: {name}", file=sys.stderr)
            continue
        try:
            for h in fn(seen, state):
                all_hits.append((name, h))
        except Exception as e:
            print(f"[warn] Check {name} abgebrochen: {e}", file=sys.stderr)

    all_hits.sort(key=lambda x: -x[1][4])  # wichtigstes zuerst

    for check, (key, title, body, url, score) in all_hits:
        seen.add(key)
        if not init and not dry:
            notify(f"[{check}] {title}", body, url,
                   "high" if score >= 8 else "default")
        elif dry:
            print(f"\n### [{check}] {title}\n{body}\n{url}")

    state["seen"] = sorted(seen)[-3000:]
    state["names"] = sorted(set(state.get("names", [])))[-500:]
    if not dry:
        save_state(state)

    if init:
        print(f"Zustand gesetzt: {len(all_hits)} Signale gemerkt, {len(seen)} Keys.")
    elif not all_hits:
        print("Keine neuen Vorlaufsignale.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", action="store_true", help="nur Zustand setzen")
    p.add_argument("--only", help="Checks kommagetrennt: " + ",".join(CHECKS))
    p.add_argument("--min-score", type=int, help="Schwelle fuer framework (Default 5)")
    p.add_argument("--dry-run", action="store_true", help="nichts senden/speichern")
    a = p.parse_args()
    if a.min_score is not None:
        MIN_SCORE = a.min_score
    run(a.init, a.only, a.dry_run)
