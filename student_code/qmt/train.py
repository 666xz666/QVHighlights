#!/usr/bin/env python3
"""Train QMT-DETR on the course split and select by dev mAP@3."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "upstream" / "moment_detr"), str(ROOT)]

from evaluate import evaluate
from locator_common import (audit_features, discover_features, dump_json, make_dataset,
                            model_config, predict)
from prepare_course_data import assert_disjoint, read_jsonl, source_id, write_jsonl
from student_code.qmt.model import build_qmt_model


def build_lr_scheduler(optimizer, epochs):
    """Build the same 75%-budget StepLR schedule as the course A0 trainer."""
    import torch

    step_size = max(1, int(epochs * 0.75))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)


def make_cyclic_mismatch(model_inputs):
    """Return a batch with each video paired to the preceding query.

    ``None`` is returned for a singleton batch because it has no in-batch negative.
    Video tensors and all non-text inputs retain their original objects.
    """
    if model_inputs["src_txt"].shape[0] < 2:
        return None
    mismatched = dict(model_inputs)
    mismatched["src_txt"] = model_inputs["src_txt"].roll(1, dims=0)
    mismatched["src_txt_mask"] = model_inputs["src_txt_mask"].roll(1, dims=0)
    return mismatched


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--dev-annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True, choices=[42, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--qmt-cross-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qmt-multiscale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qmt-hard-negative", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qmt-num-heads", type=int, default=8)
    parser.add_argument("--qmt-dilations", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--qmt-gate-bias", type=float, default=-2.0)
    parser.add_argument("--qmt-hard-negative-coef", type=float, default=0.1)
    parser.add_argument("--qmt-hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--qmt-negative-query", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qmt-negative-query-coef", type=float, default=0.1)
    parser.add_argument("--qmt-negative-query-margin", type=float, default=0.2)
    parser.add_argument("--qmt-trcc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qmt-trcc-temperature", type=float, default=0.02)
    parser.add_argument("--qmt-trcc-context-scale", type=float, default=2.0)
    parser.add_argument("--qmt-bridge", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qmt-bridge-coef", type=float, default=0.1)
    parser.add_argument("--qmt-bridge-temperature", type=float, default=0.07)
    parser.add_argument("--qmt-iou-rank", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qmt-iou-rank-coef", type=float, default=0.2)
    parser.add_argument("--qmt-iou-rank-margin", type=float, default=0.1)
    parser.add_argument("--qmt-iou-pair-coef", type=float, default=0.5)
    parser.add_argument("--qmt-iou-score-weight", type=float, default=0.5)
    return parser.parse_args()


def qmt_model_config(args):
    """Return the complete reloadable architecture configuration."""
    config = model_config(args.device)
    config.update(
        qmt_cross_attn=args.qmt_cross_attn,
        qmt_multiscale=args.qmt_multiscale,
        qmt_hard_negative=args.qmt_hard_negative,
        qmt_num_heads=args.qmt_num_heads,
        qmt_dilations=tuple(args.qmt_dilations),
        qmt_gate_bias=args.qmt_gate_bias,
        qmt_hard_negative_coef=args.qmt_hard_negative_coef,
        qmt_hard_negative_margin=args.qmt_hard_negative_margin,
        qmt_negative_query=args.qmt_negative_query,
        qmt_negative_query_coef=args.qmt_negative_query_coef,
        qmt_negative_query_margin=args.qmt_negative_query_margin,
        qmt_trcc=args.qmt_trcc,
        qmt_trcc_temperature=args.qmt_trcc_temperature,
        qmt_trcc_context_scale=args.qmt_trcc_context_scale,
        qmt_bridge=args.qmt_bridge,
        qmt_bridge_coef=args.qmt_bridge_coef,
        qmt_bridge_temperature=args.qmt_bridge_temperature,
        qmt_iou_rank=args.qmt_iou_rank,
        qmt_iou_rank_coef=args.qmt_iou_rank_coef,
        qmt_iou_rank_margin=args.qmt_iou_rank_margin,
        qmt_iou_pair_coef=args.qmt_iou_pair_coef,
        qmt_iou_score_weight=args.qmt_iou_score_weight,
    )
    return config


def evaluate_dev_predictions(predictions, dev_rows):
    """按课程评测器的行列表契约计算 dev 指标与选模 mAP。

    Args:
        predictions (list[dict]): 每个查询恰含三个候选窗口的预测行。
        dev_rows (list[dict]): dev 标注行，与预测的 qid 集合一致。

    Returns:
        tuple[dict, float]: 完整评测结果与用于 checkpoint 选择的 MR-full-mAP。
    """
    metrics = evaluate(predictions, dev_rows)
    return metrics, float(metrics["brief"]["MR-full-mAP"])


def main():
    args = parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from moment_detr.start_end_dataset import start_end_collate, prepare_batch_inputs

    if min(args.epochs, args.batch_size, args.workers + 1, args.eval_every) < 1:
        raise ValueError("epochs/batch-size/workers/eval-every must be valid")
    if (args.qmt_multiscale or args.qmt_trcc or args.qmt_bridge or args.qmt_iou_rank) and not args.qmt_cross_attn:
        raise ValueError("QMT refinements require --qmt-cross-attn")
    if args.qmt_bridge_temperature <= 0:
        raise ValueError("--qmt-bridge-temperature must be positive")
    if args.qmt_iou_rank_coef < 0 or args.qmt_iou_pair_coef < 0:
        raise ValueError("IoU ranking loss coefficients must be non-negative")
    if args.qmt_iou_score_weight < 0:
        raise ValueError("--qmt-iou-score-weight must be non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_rows, dev_rows = read_jsonl(args.train_annotations), read_jsonl(args.dev_annotations)
    assert_disjoint({"train": train_rows, "dev": dev_rows})
    layout = discover_features(args.feature_root)
    audit_features(layout, train_rows + dev_rows)
    config = qmt_model_config(args)
    train_set = make_dataset(args.train_annotations, layout, config, load_labels=True)
    dev_set = make_dataset(args.dev_annotations, layout, config, load_labels=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=start_end_collate, pin_memory=device.type == "cuda", generator=generator,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        collate_fn=start_end_collate, pin_memory=device.type == "cuda",
    )
    model, criterion = build_qmt_model(SimpleNamespace(**config))
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args.epochs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_qids = sorted(row["qid"] for row in train_rows)
    train_source_ids = sorted({source_id(row["vid"]) for row in train_rows})
    best_map, best_epoch = float("-inf"), None
    log_path = output_dir / "train_log.jsonl"

    def save_checkpoint(path, epoch, metrics):
        torch.save({
            "schema_version": 2,
            "model_family": "qmt_detr",
            "model": model.state_dict(), "epoch": epoch, "metrics": metrics,
            "model_config": config, "seed": args.seed, "train_qids": train_qids,
            "train_source_ids": train_source_ids,
        }, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        epoch_losses = []
        epoch_negative_query_losses = []
        start = time.time()
        for batch in train_loader:
            model_inputs, targets = prepare_batch_inputs(batch[1], device, non_blocking=True)
            outputs = model(**model_inputs)
            loss_dict = criterion(outputs, targets)
            mismatched_inputs = make_cyclic_mismatch(model_inputs) if args.qmt_negative_query else None
            if mismatched_inputs is not None:
                negative_outputs = model(**mismatched_inputs)
                loss_dict["loss_qmt_negative_query"] = criterion.loss_qmt_negative_query(
                    outputs, negative_outputs, targets,
                )
                epoch_negative_query_losses.append(
                    float(loss_dict["loss_qmt_negative_query"].detach()),
                )
            loss = sum(loss_dict[key] * criterion.weight_dict[key]
                       for key in loss_dict if key in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {float(loss)}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 0.1)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        record = {"epoch": epoch, "train_loss": sum(epoch_losses) / len(epoch_losses),
                  "learning_rate": epoch_learning_rate,
                  "seconds": time.time() - start}
        if epoch_negative_query_losses:
            record["train_loss_qmt_negative_query"] = (
                sum(epoch_negative_query_losses) / len(epoch_negative_query_losses)
            )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            prediction_path = output_dir / f"dev_predictions_epoch_{epoch}.jsonl"
            predictions = predict(model, dev_loader, device, "dev")
            write_jsonl(prediction_path, predictions)
            metrics, current_map = evaluate_dev_predictions(predictions, dev_rows)
            dump_json(output_dir / f"dev_metrics_epoch_{epoch}.json", metrics)
            record["dev"] = metrics
            if current_map > best_map:
                best_map, best_epoch = current_map, epoch
                save_checkpoint(output_dir / "model_best.pt", epoch, metrics)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    dump_json(output_dir / "summary.json", {
        "seed": args.seed, "best_epoch": best_epoch, "best_dev_mAP": best_map,
        "train_queries": len(train_rows), "dev_queries": len(dev_rows),
        "train_annotation_sha256": hashlib.sha256(Path(args.train_annotations).read_bytes()).hexdigest(),
        "dev_annotation_sha256": hashlib.sha256(Path(args.dev_annotations).read_bytes()).hexdigest(),
        "training_config": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.lr, "weight_decay": args.weight_decay,
            "eval_every": args.eval_every,
            "lr_scheduler": "StepLR", "lr_drop_epoch": max(1, int(args.epochs * 0.75)),
            "lr_gamma": 0.1,
        },
        "model_config": config,
    })


if __name__ == "__main__":
    main()
