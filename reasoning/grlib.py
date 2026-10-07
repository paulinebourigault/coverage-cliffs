"""Shared utilities for the reasoning-budget study.

Config loading, run-directory layout, answer extraction and grading, and the request runner
(one vLLM subprocess per model id, resumable).
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "configs", "open.yaml")
sys.path.insert(0, os.path.join(HERE, "..", "common"))  # certs.py (shared certificates)


# ----------------------------------------------------------------------------------------
# config / paths
# ----------------------------------------------------------------------------------------
def load_config(path=None):
    import yaml
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    names = [a["name"] for a in cfg["actions"]]
    assert len(set(names)) == len(names), "duplicate action names"
    for a in cfg["actions"]:
        a.setdefault("reasoning", False)
        a.setdefault("chat_template_kwargs", {})
        a.setdefault("sampling", {})
    assert cfg["reference_action"] in names
    return cfg


def action_max_tokens(cfg, a):
    return int(a.get("max_tokens", cfg["max_tokens"]))


def add_common_args(ap):
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-dir", default=None, help="default: runs/<run_name> next to the scripts")
    return ap


def run_dir(cfg, args):
    d = args.run_dir or os.path.join(HERE, "runs", cfg["run_name"])
    os.makedirs(d, exist_ok=True)
    return d


def action_names(cfg):
    return [a["name"] for a in cfg["actions"]]


def stable_hash(*parts, mod=2 ** 63 - 1):
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16) % mod


def write_parquet(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written part


def read_parts(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def wilson(k, n, z=1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ph = k / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ----------------------------------------------------------------------------------------
# answer extraction / verification
# ----------------------------------------------------------------------------------------
def last_boxed(s):
    """Content of the last \\boxed{...} / \\fbox{...} with brace matching (None if absent)."""
    if s is None:
        return None
    idx = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if idx < 0:
        return None
    i = s.find("{", idx)
    if i < 0:  # "\boxed 5"
        m = re.match(r"\\boxed\s+([^\s$]+)", s[idx:])
        return m.group(1) if m else None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j]
    return None


def answer_region(text, reasoning):
    """Strip the thinking block. If a reasoning action never closed </think> (e.g. truncated
    while thinking), there is no answer."""
    if "</think>" in text:
        return text.split("</think>")[-1]
    if reasoning or "<think>" in text:
        return ""
    return text


_NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def _to_float(s):
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s).strip()
    m = re.fullmatch(r"-?\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        v = float(m.group(1)) / float(m.group(2))
        return -v if s.startswith("-") else v
    try:
        return float(s)
    except ValueError:
        nums = _NUM.findall(s)
        if len(nums) == 1:
            try:
                return float(nums[0].replace(",", ""))
            except ValueError:
                return None
    return None


def extract_answer(text, dataset, reasoning):
    region = answer_region(text, reasoning)
    b = last_boxed(region)
    if b is not None:
        return b.strip()
    if dataset == "gsm8k":  # fallback: last number in the answer region
        nums = _NUM.findall(region)
        return nums[-1].replace(",", "") if nums else None
    return None


# --- Hendrycks-style normalizer (fallback when math_verify is unavailable) --------------
def _fix_fracs(s):
    subs = s.split("\\frac")
    out = subs[0]
    for sub in subs[1:]:
        out += "\\frac"
        if sub and sub[0] == "{":
            out += sub
        elif len(sub) >= 2:
            a, b = sub[0], sub[1]
            out += "{" + a + "}{" + b + "}" + sub[2:] if b != "{" else "{" + a + "}" + b + sub[2:]
        else:
            out += sub
    return out


def _fix_sqrt(s):
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def strip_string(s):
    s = str(s).replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    if "=" in s and len(s.split("=")[0]) <= 2:  # "x = 5" -> "5"
        s = s.split("=")[-1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        s = "\\frac{%s}{%s}" % (m.group(1), m.group(2))
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)  # thousands separators
    return s


def _sympy_equal(a, b):
    # guard: parse_expr evaluates powers eagerly (e.g. 9^{9^{9}} hangs); only try short, power-free strings
    if len(a) + len(b) > 120 or "^" in a + b or "**" in a + b:
        return False
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr

        def conv(x):
            x = x.replace("\\frac{", "(").replace("}{", ")/(").replace("}", ")").replace("{", "(")
            x = x.replace("\\sqrt", "sqrt").replace("\\pi", "pi").replace("^", "**").replace("\\cdot", "*")
            return parse_expr(x)
        return bool(sympy.simplify(conv(a) - conv(b)) == 0)
    except Exception:
        return False


try:  # optional; preferred grader for MATH
    from math_verify import parse as _mv_parse, verify as _mv_verify
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


def is_correct(pred, gold, dataset):
    if pred is None or gold is None:
        return 0
    pred, gold = str(pred).strip(), str(gold).strip()
    if dataset == "gsm8k":
        a, b = _to_float(pred), _to_float(gold)
        if a is not None and b is not None:
            return int(abs(a - b) <= 1e-6 * max(1.0, abs(b)))
        return int(strip_string(pred) == strip_string(gold))
    if strip_string(pred) == strip_string(gold):
        return 1
    if HAVE_MATH_VERIFY:
        try:
            return int(bool(_mv_verify(_mv_parse("\\boxed{" + gold + "}"),
                                       _mv_parse("\\boxed{" + pred + "}"))))
        except Exception:
            return 0  # never fall through to unbounded sympy on model output
    a, b = _to_float(strip_string(pred)), _to_float(strip_string(gold))
    if a is not None and b is not None:
        return int(abs(a - b) <= 1e-6 * max(1.0, abs(b)))
    return int(_sympy_equal(strip_string(pred), strip_string(gold)))


# ----------------------------------------------------------------------------------------
# request runner
# ----------------------------------------------------------------------------------------
REQ_COLS = ["req_id", "prompt_id", "dataset", "action", "problem", "gold"]
DEFAULT_EST_OUT = {False: 400, True: 2500}


def done_req_ids(out_dir):
    df = read_parts(os.path.join(out_dir, "part-*.parquet"))
    return set() if df.empty else set(df.req_id)


def grade_rows(rows, acts):
    for r in rows:
        a = acts[r["action"]]
        r["extracted_answer"] = extract_answer(r.pop("_text_for_grading"), r["dataset"], a["reasoning"])
        r["correct"] = is_correct(r["extracted_answer"], r["gold"], r["dataset"])
    return rows


def estimate_table(req, cfg, obs):
    """Per-action output-token estimate for pending requests. Means come from already generated
    rows (e.g. a --limit 20 pilot) when available, else from the action's `est_output_tokens`."""
    acts = {a["name"]: a for a in cfg["actions"]}
    rows = []
    for name, g in req.groupby("action", sort=False):
        a = acts[name]
        o = obs[obs.action == name] if len(obs) else obs
        if len(o):
            e_out, src = o.output_tokens.mean(), f"observed n={len(o)}"
        else:
            e_out, src = a.get("est_output_tokens", DEFAULT_EST_OUT[bool(a["reasoning"])]), "prior"
        rows.append(dict(action=name, model=a["model"], n=len(g), est_out=e_out, src=src,
                         est_Mtok_out=len(g) * e_out / 1e6))
    return pd.DataFrame(rows)


def run_requests(req, cfg, out_dir, args, salt="gen", obs_glob=None):
    """Run requests (DataFrame with REQ_COLS) -> parquet parts in out_dir, one model at a time.
    Resumable: requests whose req_id is already in out_dir are skipped. Returns False on --dry-run."""
    os.makedirs(out_dir, exist_ok=True)
    acts = {a["name"]: a for a in cfg["actions"]}
    done = done_req_ids(out_dir)
    req = req[~req.req_id.isin(done)].reset_index(drop=True)
    print(f"{len(req)} requests pending ({len(done)} already done in {out_dir})")
    if len(req) == 0:
        return True
    obs = read_parts(obs_glob or os.path.join(out_dir, "part-*.parquet"))
    est = estimate_table(req, cfg, obs)
    print("\nOUTPUT-TOKEN ESTIMATE for pending requests:")
    print(est.round({"est_out": 0, "est_Mtok_out": 3}).to_string(index=False))
    print(f"total est. output tokens {est.est_Mtok_out.sum():.2f}M\n")
    if args.dry_run:
        print("--dry-run: nothing generated.")
        return False
    for mid in dict.fromkeys(acts[a]["model"] for a in req.action):
        g = req[[acts[a]["model"] == mid for a in req.action]]
        _run_vllm(g, cfg, out_dir, args, salt, mid)
    return True


def _run_vllm(req, cfg, out_dir, args, salt, mid):
    req_path = os.path.join(out_dir, "requests.parquet")
    write_parquet(req[REQ_COLS], req_path)
    cmd = [sys.executable, os.path.join(HERE, "gen_worker.py"), "--config", args.config,
           "--requests", req_path, "--out-dir", out_dir, "--model", mid,
           "--tensor-parallel-size", str(args.tensor_parallel_size),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--max-num-seqs", str(args.max_num_seqs), "--chunk-size", str(args.chunk_size),
           "--seed", str(stable_hash(salt, cfg["seed"], mid) % 2 ** 31)]
    if args.save_text:
        cmd.append("--save-text")
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    print("\n=== model", mid, "===\n", " ".join(cmd), flush=True)
    # a fresh process per model guarantees the GPU is fully released between models
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"gen_worker failed for {mid} (exit {rc}); rerun to resume")


def add_engine_args(ap):
    ap.add_argument("--dry-run", action="store_true", help="build requests, print a token estimate, generate nothing")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=None, help="default: config max_model_len")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=4096,
                    help="requests per vLLM call (= checkpoint granularity)")
    ap.add_argument("--save-text", action=argparse.BooleanOptionalAction, default=True,
                    help="store raw generations (zstd parquet) so answers can be regraded on CPU later")
    return ap
