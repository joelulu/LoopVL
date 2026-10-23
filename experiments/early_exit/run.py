"""Shared CLI for the probe and draft_verify entrypoints."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

from .core import draft_verify, summarize
from .readout_metrics import summarize_readouts


def load_samples(path: Path, image_root: Path | None, limit: int):
    result, seen = [], set()
    root = image_root or path.resolve().parent
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id", line_number))
            if sample_id in seen:
                raise ValueError(f"Duplicate sample id: {sample_id}")
            seen.add(sample_id)
            if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
                raise ValueError(f"{sample_id}: prompt must be nonempty text")
            image = Path(row["image"]).expanduser()
            if not image.is_absolute():
                image = root / image
            if not image.is_file():
                raise FileNotFoundError(image)
            category = str(row.get("category", "unspecified"))
            if category == "__all__":
                raise ValueError("category __all__ is reserved for aggregate results")
            result.append({**row, "id": sample_id, "image": str(image.resolve()), "category": category})
            if limit and len(result) >= limit:
                break
    if not result:
        raise ValueError("Input contains no samples")
    return result


def main(mode: str):
    parser = argparse.ArgumentParser(description="LoopVL H1/H2 training-free diagnostics; NOT a speed benchmark")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, type=Path, help="JSONL: id, image, prompt, optional category")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory (must not exist)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=8, help="0 means all samples")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--logit-chunk-size", type=int, default=8)
    parser.add_argument("--check-h1-positions", type=int, default=4, help="H1-only prefix checks per sample; 0 disables")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--allow-replay-mismatch", action="store_true", help="Continue diagnostics; still report mismatches")
    if mode == "draft_verify":
        parser.add_argument("--draft-lengths", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    if args.check_h1_positions < 0 or args.limit < 0 or args.max_new_tokens < 1 or args.logit_chunk_size < 1:
        parser.error("Require limit >= 0 and positive token/chunk budgets")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("confidence threshold must be in [0,1]")
    if mode == "draft_verify" and (min(args.draft_lengths) < 1 or len(set(args.draft_lengths)) != len(args.draft_lengths)):
        parser.error("draft lengths must be positive and unique")
    samples = load_samples(args.input, args.image_root, args.limit)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(mode=mode, revision=revision, selected_samples=len(samples),
                    timing_scope="diagnostic recomputation only; no speedup claim", use_cache=False)
    (args.output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    from .runtime import LoopVLProbe
    runtime = LoopVLProbe(args.model_dir, args.device)
    settings.update(torch_version=runtime.torch.__version__, device_name=(runtime.torch.cuda.get_device_name(runtime.device) if runtime.device.type == "cuda" else "cpu"))
    (args.output / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows, generations, outcomes = [], [], []
    with (args.output / "generations.jsonl").open("w", encoding="utf-8") as gf, (args.output / "tokens.csv").open("w", newline="", encoding="utf-8") as tf, (args.output / "draft_verify.jsonl").open("w", encoding="utf-8") as df:
        writer = None
        for number, sample in enumerate(samples, 1):
            print(f"[{number}/{len(samples)}] {sample['id']}", flush=True)
            runtime.prepare(Path(sample["image"]), sample["prompt"])
            runtime.synchronize()
            started = time.perf_counter()
            reference = runtime.reference(args.max_new_tokens)
            runtime.synchronize()
            generation = {**sample, "token_ids": reference,
                          "text": runtime.tokenizer.decode(reference, skip_special_tokens=True),
                          "reference_generation_seconds": time.perf_counter()-started,
                          "visual_tokens": int(runtime.batch["visual_token_counts"][0])}
            gf.write(json.dumps(generation, ensure_ascii=False)+"\n"); gf.flush()
            generations.append(generation)
            rows = [{"id": sample["id"], "category": sample["category"], **r} for r in runtime.probe(reference, chunk_size=args.logit_chunk_size, threshold=args.confidence_threshold)]
            if rows:
                if writer is None:
                    writer = csv.DictWriter(tf, fieldnames=list(rows[0])); writer.writeheader()
                writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k,v in row.items()} for row in rows); tf.flush()
            checks = runtime.check_h1_prefixes(reference, rows, args.check_h1_positions)
            with (args.output / "h1_prefix_checks.jsonl").open("a", encoding="utf-8") as cf:
                cf.write(json.dumps({"id": sample["id"], "checks": checks}, ensure_ascii=False)+"\n")
            all_rows.extend(rows)
            (args.output / "readout_summary.json").write_text(
                json.dumps(summarize_readouts(all_rows), ensure_ascii=False, indent=2), encoding="utf-8")
            if any(not check["matches"] for check in checks) and not args.allow_replay_mismatch:
                raise RuntimeError(f"{sample['id']}: H1-only versus replay mismatch; inspect h1_prefix_checks.jsonl (masking or BF16 shape effects)")
            mismatches = sum(r["h2_id"] != r["reference_id"] for r in rows)
            summary = summarize(all_rows, generations)
            (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            if mismatches and not args.allow_replay_mismatch:
                raise RuntimeError(f"{sample['id']}: {mismatches} H2 replay mismatches; inspect tokens.csv before proceeding")
            if mode == "draft_verify":
                for length in args.draft_lengths:
                    runtime.synchronize(); started = time.perf_counter()
                    tokens, rounds = draft_verify(runtime.propose, runtime.verify, block_size=length,
                                                 max_new_tokens=args.max_new_tokens, eos_token_id=runtime.tokenizer.eos_token_id)
                    runtime.synchronize()
                    result = {"id": sample["id"], "category": sample["category"], "draft_length": length,
                              "token_ids": tokens, "text": runtime.tokenizer.decode(tokens, skip_special_tokens=True),
                              "matches_reference": tokens == reference, "rounds": rounds,
                              "diagnostic_seconds": time.perf_counter()-started,
                              "proposed_tokens": sum(len(r["draft_ids"]) for r in rounds),
                              "accepted_draft_tokens": sum(r["accepted_draft_tokens"] for r in rounds),
                              "tokens_per_verification": len(tokens)/len(rounds)}
                    df.write(json.dumps(result, ensure_ascii=False)+"\n"); df.flush(); outcomes.append(result)
                    compact = [{k:v for k,v in r.items() if k not in ("rounds", "token_ids", "text")} for r in outcomes]
                    (args.output / "draft_summary.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
                    if tokens != reference:
                        raise RuntimeError(f"{sample['id']}: draft length {length} diverged from official greedy output")
        (args.output / "COMPLETE.json").write_text(json.dumps({"samples": len(generations), "mode": mode}), encoding="utf-8")
    print(json.dumps(summarize_readouts(all_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main("probe")
