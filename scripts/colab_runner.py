"""Colab execution script for pilot gate and confirmatory GSM8K run.

Runs as a background daemon process on the Colab VM.
Resumable, records progress incrementally, and logs token counts and timing.
"""
import os
import json
import time
import traceback
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
R = Path("/content/sc-clean")
R.mkdir(parents=True, exist_ok=True)

SUFFIX = "\n\nSolve briefly step by step. Conclude your answer on a new line with: The answer is [number]."

def pack_output(x):
    return {
        "text": x.text,
        "tokens": len(x.token_ids),
        "finish_reason": x.finish_reason
    }

def main():
    import torch
    import importlib.metadata
    from vllm import LLM, SamplingParams
    import sys
    sys.path.append(str(R))
    from parser import parse_answer

    print(f"[{time.strftime('%X')}] Starting LLM initialization on {torch.cuda.get_device_name(0)}...", flush=True)
    t0 = time.time()
    
    llm = LLM(
        model="Qwen/Qwen3-1.7B",
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=256,
        gpu_memory_utilization=0.85,
        seed=35,
        disable_log_stats=True
    )
    tok = llm.get_tokenizer()
    
    model_commit = None
    try:
        model_commit = getattr(llm.llm_engine.model_config.hf_config, "_commit_hash", None)
    except Exception:
        pass

    metadata = {
        "model": "Qwen/Qwen3-1.7B",
        "gpu": torch.cuda.get_device_name(0),
        "suffix": SUFFIX,
        "k_samples": 8,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["vllm", "torch", "transformers", "datasets"]
        },
        "start_time": t0,
        "model_commit": model_commit
    }
    (R / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[{time.strftime('%X')}] Model loaded in {time.time() - t0:.1f}s", flush=True)

    # ==========================================
    # STAGE 1: PILOT GATE (50 items)
    # ==========================================
    pilot_file = R / "pilot_questions.json"
    pilot_res_file = R / "pilot_results.json"
    
    if not pilot_res_file.exists():
        print(f"[{time.strftime('%X')}] Running Pilot gate (50 items)...", flush=True)
        pilot_data = json.loads(pilot_file.read_text())
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q["question"] + SUFFIX}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for q in pilot_data
        ]
        
        # Greedy
        t_pilot = time.time()
        g_outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=512),
            use_tqdm=False
        )
        # Stochastic k=8
        s_outputs = llm.generate(
            prompts,
            [SamplingParams(n=8, temperature=0.7, top_p=0.95, top_k=-1, max_tokens=512, seed=35000 + q["pilot_id"]) for q in pilot_data],
            use_tqdm=False
        )
        
        pilot_results = []
        greedy_parse_fails = 0
        stochastic_parse_fails = 0
        total_stochastic_samples = 0
        
        for q, g, s in zip(pilot_data, g_outputs, s_outputs):
            g_packed = pack_output(g.outputs[0])
            g_parsed = parse_answer(g_packed["text"])
            if g_parsed is None:
                greedy_parse_fails += 1
                
            s_packed_list = []
            for out in s.outputs:
                sp = pack_output(out)
                parsed = parse_answer(sp["text"])
                total_stochastic_samples += 1
                if parsed is None:
                    stochastic_parse_fails += 1
                s_packed_list.append({**sp, "parsed": parsed})
                
            pilot_results.append({
                **q,
                "greedy": {**g_packed, "parsed": g_parsed},
                "samples": s_packed_list
            })
            
        greedy_fail_rate = greedy_parse_fails / len(pilot_data)
        stoch_fail_rate = stochastic_parse_fails / total_stochastic_samples
        
        pilot_summary = {
            "n_items": len(pilot_data),
            "greedy_parse_fails": greedy_parse_fails,
            "greedy_parse_fail_rate": greedy_fail_rate,
            "stochastic_parse_fails": stochastic_parse_fails,
            "total_stochastic_samples": total_stochastic_samples,
            "stochastic_parse_fail_rate": stoch_fail_rate,
            "pilot_duration_seconds": time.time() - t_pilot,
            "passed": (greedy_fail_rate <= 0.02 and stoch_fail_rate <= 0.02)
        }
        
        (R / "pilot_summary.json").write_text(json.dumps(pilot_summary, indent=2))
        pilot_res_file.write_text(json.dumps(pilot_results, indent=2))
        
        print(f"[{time.strftime('%X')}] Pilot complete: greedy parse fail rate = {greedy_fail_rate:.1%}, stoch fail rate = {stoch_fail_rate:.1%}", flush=True)
        if not pilot_summary["passed"]:
            raise RuntimeError(f"Pilot failed gate: parse failure rate too high (greedy: {greedy_fail_rate:.1%}, stoch: {stoch_fail_rate:.1%})")
    else:
        print(f"[{time.strftime('%X')}] Pilot already completed and passed.", flush=True)

    # ==========================================
    # STAGE 2: CONFIRMATORY EVALUATION (500 items)
    # ==========================================
    conf_file = R / "confirmatory_questions.json"
    conf_data = json.loads(conf_file.read_text())
    total_q = len(conf_data)
    
    print(f"[{time.strftime('%X')}] Starting confirmatory run on {total_q} fresh GSM8K items...", flush=True)
    
    for start in range(0, total_q, 50):
        dest = R / f"b{start:04d}.jsonl"
        existing = []
        if dest.exists():
            for line in dest.read_text().splitlines():
                if line.strip():
                    try:
                        existing.append(json.loads(line))
                    except json.JSONDecodeError:
                        break
            dest.write_text("".join(json.dumps(r) + "\n" for r in existing))
            
        remaining_count = 50 - len(existing)
        if remaining_count <= 0:
            print(f"[{time.strftime('%X')}] Batch {start:04d} already finished ({len(existing)} items).", flush=True)
            continue
            
        batch = conf_data[start + len(existing) : start + 50]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": q["question"] + SUFFIX}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for q in batch
        ]
        
        t_batch = time.time()
        # Greedy
        g_batch = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=512),
            use_tqdm=False
        )
        # Stochastic k=8
        s_batch = llm.generate(
            prompts,
            [SamplingParams(n=8, temperature=0.7, top_p=0.95, top_k=-1, max_tokens=512, seed=35000 + q["confirmatory_id"]) for q in batch],
            use_tqdm=False
        )
        batch_sec = time.time() - t_batch
        
        with dest.open("a") as f:
            for q, g, s in zip(batch, g_batch, s_batch):
                g_packed = pack_output(g.outputs[0])
                s_packed = [pack_output(out) for out in s.outputs]
                row = {
                    **q,
                    "greedy": g_packed,
                    "samples": s_packed,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 512,
                    "thinking": False,
                    "seed": 35000 + q["confirmatory_id"],
                    "batch_seconds": batch_sec
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                os.fsync(f.fileno())
                
        print(f"[{time.strftime('%X')}] Completed batch {start:04d}..{start+len(batch):04d} in {batch_sec:.1f}s", flush=True)
        (R / "progress.json").write_text(json.dumps({
            "completed": start + len(batch),
            "total": total_q,
            "timestamp": time.time()
        }))
        
    (R / "complete.txt").write_text(str(time.time()))
    print(f"[{time.strftime('%X')}] Confirmatory run fully completed in {time.time() - t0:.1f}s total!", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print("ERROR IN MAIN:", err, flush=True)
        (R / "error.txt").write_text(err)
        raise
