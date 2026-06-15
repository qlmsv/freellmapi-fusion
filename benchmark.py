import argparse, asyncio, json, os, random, re, subprocess, sys, tempfile, time
import httpx

UPSTREAM_BASE_URL = os.environ.get("UPSTREAM_BASE_URL", "").rstrip("/")
UPSTREAM_API_KEY  = os.environ.get("UPSTREAM_API_KEY", "")
PANEL_MODELS = [m.strip() for m in os.environ.get("PANEL_MODELS", "").split(",") if m.strip()]
JUDGE_MODELS = [m.strip() for m in os.environ.get("JUDGE_MODEL", "").split(",") if m.strip()]
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "4"))
TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))

JUDGE_SYSTEM = ("You are the synthesizer in a model-fusion system. Several AI models each "
  "answered the same request. Find consensus, resolve contradictions, keep unique insights, "
  "cover blind spots, then write ONE best final answer. Keep the required answer format.")

async def call(client, model, messages, sem, retries=2):
    payload = {"model": model, "messages": messages, "stream": False}
    headers = {"Authorization": f"Bearer {UPSTREAM_API_KEY}", "Content-Type": "application/json"}
    async with sem:
        for attempt in range(retries + 1):
            try:
                r = await client.post(f"{UPSTREAM_BASE_URL}/chat/completions", json=payload,
                                      headers=headers, timeout=TIMEOUT)
                if r.status_code == 429 and attempt < retries:
                    await asyncio.sleep(2 * (attempt + 1)); continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(attempt + 1); continue
                return None

async def run_systems(client, prompt, sem):
    messages = [{"role": "user", "content": prompt}]
    panel_answers = await asyncio.gather(*[call(client, m, messages, sem) for m in PANEL_MODELS])
    out = {f"panel:{m}": a for m, a in zip(PANEL_MODELS, panel_answers)}
    good = [(m, a) for m, a in zip(PANEL_MODELS, panel_answers) if a and a.strip()]
    out["judge-alone"] = await call(client, JUDGE_MODELS[0], messages, sem) if JUDGE_MODELS else None
    if good:
        panel_block = "\n\n".join(f"### Model {i+1} ({m})\n{a}" for i, (m, a) in enumerate(good))
        jmsg = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"REQUEST:\n{prompt}\n\nPANEL ANSWERS:\n{panel_block}\n\nWrite the single best final answer."}]
        fusion = None
        for jm in JUDGE_MODELS:
            fusion = await call(client, jm, jmsg, sem)
            if fusion and fusion.strip():
                break
        out["FUSION"] = fusion
    else:
        out["FUSION"] = None
    return out

def gsm8k_prompt(item):
    return (item["question"] + "\n\nSolve step by step. On the last line write exactly: "
            "'Answer: <number>' with only the final number.")

def gsm8k_grade(ans, item):
    if not ans: return False
    gold = item["answer"].split("####")[-1].strip().replace(",", "")
    m = re.findall(r"-?\d[\d,]*\.?\d*", ans.replace(",", ""))
    return bool(m) and m[-1].rstrip(".").lstrip("0").rjust(1, "0") == gold.lstrip("0").rjust(1, "0")

def mc_prompt(question, choices):
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (f"{question}\n\n{body}\n\nAnswer with ONLY the single letter (A, B, C, or D) "
            "on the last line as 'Answer: <letter>'.")

def mc_extract(ans):
    if not ans: return None
    m = re.findall(r"Answer:\s*([A-D])", ans, re.I)
    if m: return m[-1].upper()
    m = re.findall(r"\b([A-D])\b", ans)
    return m[-1].upper() if m else None

FENCE = chr(96) * 3

def extract_code(ans):
    if not ans: return ""
    blocks = re.findall(FENCE + r"(?:python)?\s*(.*?)" + FENCE, ans, re.S)
    return (blocks[0] if blocks else ans).strip()

def humaneval_grade(ans, item):
    code = extract_code(ans)
    program = code + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program); path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass

def load_dataset_items(name, n, seed=0):
    from datasets import load_dataset
    rng = random.Random(seed)
    if name == "gsm8k":
        ds = list(load_dataset("gsm8k", "main", split="test"))
        rng.shuffle(ds); ds = ds[:n]
        return [{"prompt": gsm8k_prompt(x), "grade": ("gsm8k", x)} for x in ds]
    if name == "mmlu":
        ds = list(load_dataset("cais/mmlu", "all", split="test"))
        rng.shuffle(ds); ds = ds[:n]
        items = []
        for x in ds:
            gold = "ABCD"[x["answer"]]
            items.append({"prompt": mc_prompt(x["question"], x["choices"]), "grade": ("mc", gold)})
        return items
    if name == "gpqa":
        ds = list(load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                               token=os.environ.get("HF_TOKEN")))
        rng.shuffle(ds); ds = ds[:n]
        items = []
        for x in ds:
            choices = [x["Correct Answer"], x["Incorrect Answer 1"],
                       x["Incorrect Answer 2"], x["Incorrect Answer 3"]]
            order = [0, 1, 2, 3]; rng.shuffle(order)
            shuffled = [choices[i] for i in order]
            gold = "ABCD"[order.index(0)]
            items.append({"prompt": mc_prompt(x["Question"], shuffled), "grade": ("mc", gold)})
        return items
    if name == "humaneval":
        ds = list(load_dataset("openai_humaneval", split="test"))
        rng.shuffle(ds); ds = ds[:n]
        return [{"prompt": x["prompt"] + "\n\nComplete this function. Return the full function in a "
                 + FENCE + "python code block.", "grade": ("humaneval", x)} for x in ds]
    raise ValueError(name)

def grade(kind, ref, ans):
    if kind == "gsm8k":   return gsm8k_grade(ans, ref)
    if kind == "mc":      return mc_extract(ans) == ref
    if kind == "humaneval": return humaneval_grade(ans, ref)
    return False

async def run_benchmark(name, items, ckpt_path, resume):
    done = {}
    if resume and os.path.exists(ckpt_path):
        done = {r["i"]: r for r in (json.loads(l) for l in open(ckpt_path))}
        print(f"  [{name}] resuming, {len(done)} already done")
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    systems = [f"panel:{m}" for m in PANEL_MODELS] + ["judge-alone", "FUSION"]
    f = open(ckpt_path, "a")
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(items):
            if i in done: continue
            t0 = time.time()
            answers = await run_systems(client, item["prompt"], sem)
            kind, ref = item["grade"]
            row = {"i": i, "latency": round(time.time() - t0, 1),
                   "correct": {s: bool(grade(kind, ref, answers.get(s))) for s in systems}}
            f.write(json.dumps(row) + "\n"); f.flush()
            done[i] = row
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                acc = sum(r["correct"].get("FUSION", False) for r in done.values())
                print(f"  [{name}] {i+1}/{len(items)}  fusion_acc={acc}/{len(done)}")
    f.close()
    return list(done.values())

def summarize(name, rows):
    systems = [f"panel:{m}" for m in PANEL_MODELS] + ["judge-alone", "FUSION"]
    n = len(rows)
    print(f"\n=== {name}  (n={n}) ===")
    print(f"{'system':<45} {'accuracy':>10}")
    for s in systems:
        c = sum(r["correct"].get(s, False) for r in rows)
        print(f"{s:<45} {c/n*100:>8.1f}%  ({c}/{n})")
    avg_lat = sum(r["latency"] for r in rows) / n
    print(f"avg latency/question: {avg_lat:.1f}s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="gsm8k,mmlu,gpqa,humaneval")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--outdir", default="bench_results")
    args = ap.parse_args()
    if not (UPSTREAM_BASE_URL and UPSTREAM_API_KEY and PANEL_MODELS and JUDGE_MODELS):
        sys.exit("Set UPSTREAM_BASE_URL, UPSTREAM_API_KEY, PANEL_MODELS, JUDGE_MODEL first.")
    n = 10 if args.smoke else args.n
    os.makedirs(args.outdir, exist_ok=True)
    all_rows = {}
    for name in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        print(f"\nLoading {name} (up to {n})...")
        items = load_dataset_items(name, n)
        print(f"  got {len(items)} questions")
        rows = asyncio.run(run_benchmark(name, items, os.path.join(args.outdir, f"{name}.jsonl"), args.resume))
        all_rows[name] = rows
        summarize(name, rows)
    print("\n\n########## SUMMARY ##########")
    for name, rows in all_rows.items():
        summarize(name, rows)

if __name__ == "__main__":
    main()
