#!/usr/bin/env python3
"""vLLM worker: run all pending requests of ONE model id (all of its actions, e.g. thinking
on/off, mixed in the same batches), grade them, and checkpoint parquet parts. Launched by
grlib.run_requests in a fresh process per model so GPU memory is released between models.
"""
import argparse
import os
import time

import pandas as pd

import grlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-text", action="store_true")
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()

    cfg = grlib.load_config(args.config)
    assert grlib.HAVE_MATH_VERIFY, "pip install math-verify (answers are graded here)"
    acts = {a["name"]: a for a in cfg["actions"] if a["model"] == args.model}
    req = pd.read_parquet(args.requests)
    req = req[req.action.isin(acts)]
    done = grlib.done_req_ids(args.out_dir)
    req = req[~req.req_id.isin(done)].reset_index(drop=True)
    print(f"[{args.model}] actions={list(acts)} pending={len(req)} done_before={len(done)}", flush=True)
    if len(req) == 0:
        return
    # interleave actions so long thinking traces and short answers share batches
    req = req.sample(frac=1.0, random_state=args.seed % 2 ** 31).reset_index(drop=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm_kw = dict(cfg.get("llm_kwargs", {}).get("default", {}))
    llm_kw.update(cfg.get("llm_kwargs", {}).get(args.model, {}))
    max_len = args.max_model_len or cfg.get("max_model_len") or max(grlib.action_max_tokens(cfg, a) for a in acts.values()) + 2048
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs,
              max_model_len=max_len, seed=args.seed, enforce_eager=args.enforce_eager, **llm_kw)
    sps = {n: SamplingParams(max_tokens=grlib.action_max_tokens(cfg, a), n=1, **a["sampling"]) for n, a in acts.items()}

    def render(r):
        a = acts[r["action"]]
        msgs = [{"role": "user", "content": cfg["prompt_template"].format(problem=r["problem"])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **a["chat_template_kwargs"])

    tag = args.model.replace("/", "__")
    k0 = int(time.time())
    tot_out, tot_t = 0, 0.0
    for ci, s in enumerate(range(0, len(req), args.chunk_size)):
        chunk = req.iloc[s:s + args.chunk_size].to_dict("records")
        texts = [render(r) for r in chunk]
        t0 = time.time()
        outs = llm.generate(texts, [sps[r["action"]] for r in chunk], use_tqdm=True)
        dt = time.time() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        rows, txt = [], []
        for r, o in zip(chunk, outs):
            g = o.outputs[0]
            n_out = len(g.token_ids)
            rows.append(dict(req_id=r["req_id"], prompt_id=r["prompt_id"], dataset=r["dataset"],
                             action=r["action"], model=args.model, gold=r["gold"],
                             input_tokens=len(o.prompt_token_ids), output_tokens=n_out,
                             finish_reason=str(g.finish_reason),
                             think_closed=("</think>" in g.text),
                             # chunk wall time apportioned by output-token share (aggregate throughput
                             # only: per-request latency is confounded by continuous batching)
                             gen_seconds_approx=dt * n_out / max(ntok, 1),
                             n_gpus=args.tensor_parallel_size, _text_for_grading=g.text))
            if args.save_text:
                txt.append(dict(req_id=r["req_id"], text=g.text))
        grlib.grade_rows(rows, acts)
        grlib.write_parquet(pd.DataFrame(rows), os.path.join(args.out_dir, f"part-{tag}-{k0}-{ci:04d}.parquet"))
        if txt:
            os.makedirs(os.path.join(args.out_dir, "text"), exist_ok=True)
            grlib.write_parquet(pd.DataFrame(txt), os.path.join(args.out_dir, "text", f"text-{tag}-{k0}-{ci:04d}.parquet"))
        tot_out += ntok; tot_t += dt
        df = pd.DataFrame(rows)
        acc = df.groupby("action").correct.mean().round(3).to_dict()
        trunc = (df.finish_reason == "length").groupby(df.action).mean().round(3).to_dict()
        print(f"[{args.model}] chunk {ci}: {len(chunk)} reqs, {ntok} out-tokens in {dt:.0f}s "
              f"=> {ntok / dt:.0f} tok/s (cum {tot_out / tot_t:.0f} tok/s, {len(chunk) / dt:.2f} req/s) "
              f"acc={acc} trunc={trunc}", flush=True)
    print(f"[{args.model}] DONE: {tot_out} output tokens in {tot_t / 3600:.2f} h "
          f"({tot_out / max(tot_t, 1e-9):.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
