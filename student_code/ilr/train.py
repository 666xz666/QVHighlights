#!/usr/bin/env python3
"""Train pointwise QLoRA with fixed-pair IoU-aware listwise supervision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from student_code.ilr.objective import combined_ilr_loss, first_answer_score
from vlm.common import file_sha256, load_jsonl
from vlm.modeling import load_model, prepare_inputs, read_config, to_cuda
from vlm.train import snapshot_trainable, update_evidence
from vlm.video import VideoStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True,
                        help="Fixed data/reference_full/sft_pairwise.jsonl")
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, choices=[42, 2026, 3407],
                        help="Override config seed for the three formal runs")
    parser.add_argument("--disable-ilr", action="store_true",
                        help="C-off: keep pair grouping/exposure but set both auxiliary weights to zero")
    return parser.parse_args()


def validate_pairs(rows):
    """Validate the immutable two-candidate course training pool."""
    if not rows:
        raise ValueError("No training pairs")
    for row in rows:
        if row.get("split") != "train" or row.get("task") != "pairwise":
            raise ValueError("Every ILR row must be a pairwise train example")
        if len(row.get("clips", [])) != 2 or len(row.get("target_tiou", [])) != 2:
            raise ValueError("Every ILR row must contain two clips and two target_tiou values")
        winner = 0 if row["answer"] == "A" else 1 if row["answer"] == "B" else None
        if winner is None or row["target_tiou"][winner] <= row["target_tiou"][1 - winner]:
            raise ValueError("Pairwise answer must select the higher-IoU clip")


def main():
    args = parse_args()
    import torch
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    config = read_config(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.seed is not None:
        config["seed"] = args.seed
    if args.disable_ilr:
        config["listwise_weight"] = 0.0
        config["pairwise_weight"] = 0.0
    required = {"task": "pointwise", "gradient_accumulation_steps": 4}
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"ILR config requires {key}={expected!r}")
    weights = (config.get("listwise_weight"), config.get("pairwise_weight"))
    if weights not in {(0.5, 0.2), (0.0, 0.0)}:
        raise ValueError("ILR weights must be (0.5, 0.2), or (0, 0) for C-off")
    if config["max_steps"] < 1:
        raise ValueError("max_steps must be positive")
    rows = load_jsonl(args.train_data)
    validate_pairs(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "adapter_config.json").exists() and not args.resume:
        raise FileExistsError("Output already contains an adapter; use a new directory or --resume")

    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, processor, dtype = load_model(config, training=True)
    store = VideoStore(args.video_root, args.cache_dir, config["num_frames"],
                       config["cache_max_side"], args.cache_only)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    yes_ids = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = processor.tokenizer.encode("No", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        raise RuntimeError("Pinned tokenizer must encode Yes and No as one token")

    step, offset, epoch = 0, 0, 0
    order = list(range(len(rows)))
    random.shuffle(order)
    train_sha = file_sha256(args.train_data)
    if args.resume:
        checkpoint = Path(args.resume)
        previous = json.loads((checkpoint / "training_config.json").read_text())
        resume_keys = (
            "model_name", "task", "load_in_4bit", "lora_rank", "lora_targets",
            "num_frames", "max_pixels", "listwise_weight", "pairwise_weight",
            "tau_score", "tau_target", "tau_pair", "seed",
        )
        for key in resume_keys:
            if config[key] != previous[key]:
                raise ValueError(f"Resume config mismatch: {key}")
        set_peft_model_state_dict(model, load_peft_weights(str(checkpoint), device="cuda"))
        state = torch.load(checkpoint / "trainer_state.pt", map_location="cuda", weights_only=True)
        if state["train_data_sha256"] != train_sha:
            raise ValueError("Training data changed since checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, offset, epoch, order = state["step"], state["offset"], state["epoch"], state["order"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])

    run_start_step = step
    initial_trainable = snapshot_trainable(model)

    def save_checkpoint(destination):
        destination.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(destination, safe_serialization=True)
        processor.save_pretrained(destination)
        (destination / "training_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8",
        )
        torch.save({
            "step": step, "offset": offset, "epoch": epoch, "order": order,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(), "train_data_sha256": train_sha,
        }, destination / "trainer_state.pt")
        evidence = update_evidence(model, initial_trainable, optimizer, run_start_step, step)
        (destination / "parameter_update_evidence.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8",
        )

    metadata = {
        "method": "ILR-QLoRA" if weights != (0.0, 0.0) else "ILR-QLoRA-off",
        "training_pairs": len(rows), "candidates_per_update": 8,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "run_start_step": run_start_step, "config": config,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    accum = config["gradient_accumulation_steps"]
    log_path = output / "training_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    while step < config["max_steps"]:
        component_totals = {"total": 0.0, "ce": 0.0, "listwise": 0.0, "pairwise": 0.0}
        sampling_adjustments = []
        for _ in range(accum):
            if offset == len(order):
                offset = 0
                epoch += 1
                random.shuffle(order)
            row = rows[order[offset]]
            targets = torch.tensor(row["target_tiou"], dtype=torch.float32, device="cuda:0")
            ce_losses, candidate_scores = [], []
            for candidate_index, clip in enumerate(row["clips"]):
                is_winner = candidate_index == int(targets.argmax().item())
                batch = prepare_inputs(
                    processor, row["query"], [clip], "pointwise", store, row["vid"],
                    answer="Yes" if is_winner else "No", max_tokens=config["max_tokens"],
                )
                sampling_adjustments.extend(store.drain_sampling_events())
                cuda_batch = to_cuda(batch)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    model_output = model(**cuda_batch, use_cache=False)
                ce_losses.append(model_output.loss.float())
                candidate_scores.append(first_answer_score(
                    model_output.logits, cuda_batch["labels"], yes_ids[0], no_ids[0],
                ))
            scores = torch.stack(candidate_scores)
            loss, components = combined_ilr_loss(
                ce_losses, scores, targets,
                listwise_weight=config["listwise_weight"],
                pairwise_weight=config["pairwise_weight"],
                tau_score=config["tau_score"], tau_target=config["tau_target"],
                tau_pair=config["tau_pair"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}, qid {row['qid']}")
            scaler.scale(loss / accum).backward()
            component_totals["total"] += float(loss.detach())
            for key, value in components.items():
                component_totals[key] += float(value.detach())
            offset += 1
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        multiplier = min(1.0, (step + 1) / max(1, config["warmup_steps"]))
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * multiplier
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        record = {
            "step": step, "epoch": epoch,
            **{key + "_loss" if key != "total" else "loss": value / accum
               for key, value in component_totals.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "gradient_norm_before_clip": float(gradient_norm.detach()),
            "sampling_adjustments": sampling_adjustments,
            "elapsed_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if step % config["save_every"] == 0:
            save_checkpoint(output / f"checkpoint-{step}")
    save_checkpoint(output)
    print(f"Adapter update evidence: {output / 'parameter_update_evidence.json'}")
    print(f"Saved final adapter and resumable trainer state to {output}")


if __name__ == "__main__":
    main()
