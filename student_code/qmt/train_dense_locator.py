"""训练密集定位器，按 dev mAP@3 选模，并导出无标签 Top-3。"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate import evaluate, tiou
from scripts.locator_common import audit_features, discover_features, dump_json, make_dataset, model_config
from scripts.prepare_course_data import assert_disjoint, read_jsonl, write_jsonl
from moment_detr.start_end_dataset import prepare_batch_inputs, start_end_collate
from student_code.qmt.dense_locator import DenseLocator, LEVEL_STRIDES, dense_losses
from student_code.qmt.dense_rank import oracle_order, rank_windows, top_three
from student_code.qmt.resume import restore_training_state, save_training_state


def loader(annotations, layout, config, args, generator=None):
    """构造课程特征加载器；训练使用原标注中的全量 GT。"""
    dataset = make_dataset(annotations, layout, config, load_labels=False)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=generator is not None,
                      generator=generator, num_workers=args.workers,
                      collate_fn=start_end_collate, pin_memory=args.device == "cuda")


def forward(model, batch, device):
    """把真时长与当前批次特征传给密集定位器，保持秒坐标一致。"""
    metadata, tensors = batch
    inputs, _ = prepare_batch_inputs(tensors, device, non_blocking=device.type == "cuda")
    duration = inputs["src_vid"].new_tensor([row["duration"] for row in metadata])
    return model(**inputs, duration=duration)


def pack_truth(metadata, truth_by_qid, reference):
    """打包当前批次的完整秒级 GT，返回 [B G 2] 区间和 [B G] 掩码。"""
    rows = [truth_by_qid[row["qid"]] for row in metadata]
    count = max(map(len, rows))
    if not count or any(not row for row in rows):
        raise ValueError("Every training query must contain at least one GT window")
    padded = [row + [[0, 0]] * (count - len(row)) for row in rows]
    valid = [[True] * len(row) + [False] * (count - len(row)) for row in rows]
    return reference.new_tensor(padded), torch.tensor(valid, device=reference.device)


@torch.inference_mode()
def predict_ranked(model, batches, device, split):
    """为每条查询导出前十个唯一密集窗口，推理只使用特征和视频长度。"""
    model.eval()
    rows = []
    for batch in batches:
        output = forward(model, batch, device)
        scores = (output["relevance"].sigmoid() * output["quality"].sigmoid()).cpu().numpy()
        spans = output["spans"].cpu().numpy()
        valid = output["valid"].cpu().numpy()
        for meta, row_spans, row_scores, mask in zip(batch[0], spans, scores, valid, strict=True):
            row = {key: meta[key] for key in ("qid", "vid", "query", "duration")}
            ranked = rank_windows(row_spans[mask], row_scores[mask], row["duration"])
            rows.append(dict(row, split=split, pred_relevant_windows=ranked))
    return rows


def top_three_rows(ranked):
    """将密集候选列表转换为课程严格 Top-3 预测。"""
    return [dict(row, pred_relevant_windows=top_three(row["pred_relevant_windows"]))
            for row in ranked]


def shortlist_diagnostics(ranked, truth):
    """统计 Top-3/10 长度分桶 GT 覆盖率和 Top-10 oracle mAP@3。"""
    indexed = {row["qid"]: row["relevant_windows"] for row in truth}
    counts = {name: np.zeros((2, 3), dtype=int) for name in ("all", "short", "middle", "long")}
    oracle = []
    for row in ranked:
        windows = row["pred_relevant_windows"]
        gt = indexed[row["qid"]]
        oracle.append(dict(row, pred_relevant_windows=oracle_order(windows, gt)))
        for window in gt:
            width = window[1] - window[0]
            bucket = "short" if width <= 10 else "middle" if width <= 30 else "long"
            for index, cap in enumerate((3, 10)):
                iou = max(tiou(window, proposal) for proposal in windows[:cap])
                delta = np.array([1, iou >= 0.5, iou >= 0.7])
                counts["all"][index] += delta
                counts[bucket][index] += delta
    coverage = {}
    for name, records in counts.items():
        coverage[name] = {str(cap): {
            "gt": int(total), "hit_0.5": int(hit50), "hit_0.7": int(hit70),
            "coverage_0.5_pct": float(100 * hit50 / total) if total else None,
            "coverage_0.7_pct": float(100 * hit70 / total) if total else None,
        } for cap, (total, hit50, hit70) in zip((3, 10), records, strict=True)}
    return {"coverage": coverage, "oracle_top3_from_top10": evaluate(oracle, truth)}


def train(args, *, stop_after=None):
    """训练和在 epoch 边界恢复；测试可在固定预算中途主动暂停。"""
    if min(args.epochs, args.batch_size, args.eval_every, args.workers + 1) < 1 or args.lr <= 0:
        raise ValueError("Invalid training budget")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    train_rows = read_jsonl(args.train_annotations)
    dev_rows = read_jsonl(args.dev_annotations)
    assert_disjoint({"train": train_rows, "dev": dev_rows})
    truth_by_qid = {row["qid"]: row["relevant_windows"] for row in train_rows}
    layout = discover_features(args.feature_root)
    audit_features(layout, train_rows + dev_rows)
    config = model_config(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = loader(args.train_annotations, layout, config, args, gen)
    dev_loader = loader(args.dev_annotations, layout, config, args)
    model = DenseLocator(config, args.scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_drop = max(1, int(args.epochs * 0.75))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop, gamma=0.1)
    hashes = {key + "_annotation_sha256": hashlib.sha256(
        getattr(args, key + "_annotations").read_bytes()).hexdigest() for key in ("train", "dev")}
    budget = dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
                  weight_decay=1e-4, eval_every=args.eval_every, lr_drop_epoch=lr_drop, lr_gamma=0.1)
    contract = dict(scale=args.scale, seed=args.seed, workers=args.workers, config=config,
                    budget=budget, layout=layout, **hashes)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log = output_dir / "train_log.jsonl"
    state = dict(epoch=0, contract=contract, history=[], best=None, best_ranked=None)
    if args.resume:
        state = restore_training_state(args.resume, model, optimizer, scheduler, gen, contract)
        print(json.dumps({"resume_from": str(args.resume), "next_epoch": state["epoch"] + 1}), flush=True)
    write_jsonl(log, state["history"])

    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        model.train()
        start = time.monotonic()
        totals, seen = np.zeros(4), 0
        lr = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            output = forward(model, batch, device)
            truth, mask = pack_truth(batch[0], truth_by_qid, output["spans"])
            losses = dense_losses(output, truth, mask, args.scale)
            loss = losses["relevance"] + losses["l1"] + 2 * losses["giou"] + losses["quality"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite dense loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            seen += len(batch[0])
            totals += np.array([value.detach().item() for value in losses.values()]) * len(batch[0])
        scheduler.step()
        record = dict(epoch=epoch, learning_rate=lr, seconds=time.monotonic() - start,
                      losses=dict(zip(losses, (totals / seen).tolist(), strict=True)))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            ranked = predict_ranked(model, dev_loader, device, "dev")
            predictions = top_three_rows(ranked)
            metrics = evaluate(predictions, dev_rows)
            record["dev"] = metrics["brief"]
            write_jsonl(output_dir / f"dev_predictions_epoch_{epoch}.jsonl", predictions)
            dump_json(output_dir / f"dev_metrics_epoch_{epoch}.json", metrics)
            current = metrics["brief"]["MR-full-mAP"]
            if state["best"] is None or current > state["best"]["metrics"]["brief"]["MR-full-mAP"]:
                state["best"] = dict(
                    schema_version=1, model_family="dense_locator", scale=args.scale,
                    model_config=config, model={k: v.detach().cpu().clone()
                                                for k, v in model.state_dict().items()},
                    metrics=metrics, epoch=epoch, seed=args.seed, **hashes,
                )
                state["best_ranked"] = ranked
        state["epoch"] = epoch
        state["history"].append(record)
        save_training_state(output_dir / "model_last.pt", model, optimizer, scheduler, gen, state)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if stop_after == epoch:
            return

    best = state["best"]
    ranked = state["best_ranked"]
    checkpoint = output_dir / "model_best.pt"
    temporary = checkpoint.with_suffix(".tmp")
    torch.save(best, temporary)
    temporary.replace(checkpoint)
    predictions = top_three_rows(ranked)
    write_jsonl(output_dir / "dev_best_top3.jsonl", predictions)
    write_jsonl(output_dir / "dev_best_top10.jsonl", ranked)
    dump_json(output_dir / "summary.json", dict(
        seed=args.seed, scale=args.scale, strides_sec=LEVEL_STRIDES[args.scale],
        best_epoch=best["epoch"], best_dev_mAP=best["metrics"]["brief"]["MR-full-mAP"],
        metrics=best["metrics"], diagnostics=shortlist_diagnostics(ranked, dev_rows),
        train_queries=len(train_rows), dev_queries=len(dev_rows), training_config=budget, **hashes,
    ))


def export(args):
    """加载最佳密集模型，在无标签输入上导出严格课程 Top-3。"""
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (state.get("schema_version"), state.get("model_family")) != (1, "dense_locator"):
        raise ValueError("Expected a dense locator model_best.pt")
    config = dict(state["model_config"], device=args.device)
    device = torch.device(args.device)
    model = DenseLocator(config, state["scale"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    layout = discover_features(args.feature_root)
    audit_features(layout, read_jsonl(args.annotations))
    batches = loader(args.annotations, layout, config, args)
    ranked = predict_ranked(model, batches, device, args.split)
    write_jsonl(args.output, top_three_rows(ranked))


def main():
    """解析训练与无标签导出的命令行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    training = subparsers.add_parser("train")
    exporting = subparsers.add_parser("export")
    for command in (training, exporting):
        command.add_argument("--feature-root", type=Path, default=Path("data/features"))
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    training.add_argument("--scale", choices=LEVEL_STRIDES, required=True)
    training.add_argument("--seed", type=int, choices=(42, 2026, 3407), required=True)
    training.add_argument("--epochs", type=int, default=90)
    training.add_argument("--lr", type=float, default=1.5e-4)
    training.add_argument("--eval-every", type=int, default=5)
    training.add_argument("--train-annotations", type=Path, default=Path("data/course/train.jsonl"))
    training.add_argument("--dev-annotations", type=Path, default=Path("data/course/dev.jsonl"))
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.set_defaults(run=train)
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--annotations", type=Path, required=True)
    exporting.add_argument("--split", required=True)
    exporting.add_argument("--output", type=Path, required=True)
    exporting.set_defaults(run=export)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
