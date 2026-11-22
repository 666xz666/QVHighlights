"""训练或加载短窗口分支，主定位器始终冻结，导出课程格式 Top-3。"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from scripts.locator_common import discover_features, dump_json, make_dataset, sanitize_window
from scripts.evaluate import evaluate, tiou
from scripts.prepare_course_data import assert_disjoint, read_jsonl, source_id, write_jsonl
from moment_detr.span_utils import span_cxw_to_xx
from moment_detr.start_end_dataset import prepare_batch_inputs, start_end_collate
from student_code.qmt.model import build_qmt_model
from student_code.qmt.resume import restore_training_state, save_training_state
from student_code.qmt.short_windows import (
    CLIP_SECONDS, ap_gain, fit_gate, gate_features, grid_features, make_head,
    predict_gain, rank_short, replace_third, score_loss, short_targets, short_truth,
)


@dataclass
class Cached:
    """一条预测及对应有效 clip 的冻结 encoder 表示 [T D]，缓存在 CPU。"""

    prediction: dict
    memory: torch.Tensor


def sha256(path):
    """流式计算输入文件签名。"""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_backbone(path, device):
    """严格加载同一种子的多尺度动态锚点，并冻结全部权重。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = {**checkpoint["model_config"], "device": str(device)}
    if not (config["qmt_cross_attn"] and config["qmt_tcn"] and
            config["qmt_anchor_mode"] == "multiscale" and config["clip_length"] == CLIP_SECONDS):
        raise ValueError("Expected the CrossTCN multiscale-anchor backbone")
    model, _ = build_qmt_model(SimpleNamespace(**config))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).requires_grad_(False).eval(), config, checkpoint["seed"], checkpoint["epoch"]


def cache_split(backbone, config, layout, annotations, split, device, batch_size):
    """对每条查询只运行一次冻结模型，保留原三项及可用 clip 表示。"""
    loader = DataLoader(make_dataset(annotations, layout, config, load_labels=False),
                        batch_size=batch_size, collate_fn=start_end_collate, shuffle=False)
    records = []
    # 使用 no_grad 创建普通张量，训练分支反传时需要保存这些输入。
    with torch.no_grad():
        for metadata, batch in loader:
            inputs, _ = prepare_batch_inputs(batch, device)
            out = backbone(**inputs, return_video_memory=True)
            spans = span_cxw_to_xx(out["pred_spans"]).cpu().tolist()
            scores = out["pred_logits"].softmax(-1)[..., 0].cpu().tolist()
            lengths = out["video_mask"].sum(-1).cpu().tolist()
            memory = out["video_memory"].cpu()
            for i, row in enumerate(metadata):
                duration = row["duration"]
                ranked = sorted(zip(spans[i], scores[i]), key=lambda p: -p[1])[:3]
                windows = [[*sanitize_window(a * duration, b * duration, duration), round(s, 8)]
                           for (a, b), s in ranked]
                clip_count = min(lengths[i], int(duration // CLIP_SECONDS))
                prediction = {k: row[k] for k in ("qid", "vid", "query", "duration")}
                prediction.update(split=split, pred_relevant_windows=windows)
                records.append(Cached(prediction, memory[i, :clip_count].clone()))
    return records


def split_for_gate(rows):
    """按源视频分组，以固定哈希排序留出 20% 供替换门拟合。"""
    sources = {source_id(row["vid"]) for row in rows}
    if len(sources) < 2:
        raise ValueError("Need at least two training source videos")
    order = sorted(sources, key=lambda s: hashlib.sha256(s.encode()).digest())
    gate_sources = set(order[:max(1, len(order) // 5)])
    head_ids, gate_ids = [], []
    for row in rows:
        target = gate_ids if source_id(row["vid"]) in gate_sources else head_ids
        target.append(row["qid"])
    return head_ids, gate_ids


def score_batch(head, records, device):
    """从不同长度的缓存表示计算 [B W] logits 及有效窗口掩码。"""
    memory = pad_sequence([r.memory for r in records], batch_first=True).to(device)
    lengths = torch.as_tensor([len(r.memory) for r in records], device=device)
    features, windows, valid = grid_features(memory, lengths)
    return head(features).squeeze(-1), windows, valid


@torch.no_grad()
def ranked_short(head, records, device, batch_size):
    """获取每条查询过滤重复边界后的三个最高分短窗口。"""
    head.eval()
    ranked = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        logits, windows, valid = score_batch(head, batch, device)
        scores = logits.sigmoid().cpu()
        windows, valid = windows.cpu(), valid.cpu()
        ranked.extend(rank_short(windows[mask].tolist(), score[mask].tolist(),
                                 r.prediction["pred_relevant_windows"])
                      for r, score, mask in zip(batch, scores, valid))
    return ranked


def calibrate(records, ranked, rows_by_id):
    """以留出集预测窗口和全部 GT 拟合第 3 项 AP 增益。"""
    features, gains = [], []
    for record, proposals in zip(records, ranked, strict=True):
        if proposals:
            base = record.prediction["pred_relevant_windows"]
            features.append(gate_features(base, proposals))
            truth = rows_by_id[record.prediction["qid"]]["relevant_windows"]
            gains.append(ap_gain(base, proposals[0], truth))
    return fit_gate(features, gains)


def merge(records, ranked, gate):
    """把短窗口放入第 3 项，gate=None 表示总是换入的推理消融。"""
    merged = []
    replacements = 0
    for record, proposals in zip(records, ranked, strict=True):
        base = record.prediction["pred_relevant_windows"]
        accept = proposals and (gate is None or predict_gain(gate, gate_features(base, proposals)) > 0)
        windows = replace_third(base, proposals[0] if accept else None)
        replacements += windows != base
        merged.append(dict(record.prediction, pred_relevant_windows=windows))
    return merged, replacements


def short_coverage(pools, rows):
    """按 GT 窗口微平均统计短区间覆盖率，阈值为 0.5 和 0.7。"""
    total = 0
    matches = [0, 0]
    for pool, row in zip(pools, rows, strict=True):
        for truth in short_truth(row):
            total += 1
            overlap = max((tiou(proposal, truth) for proposal in pool), default=0)
            for i, threshold in enumerate((.5, .7)):
                matches[i] += overlap >= threshold
    return {"gt_windows": total, "hits@0.5": matches[0], "hits@0.7": matches[1],
            "coverage@0.5": 100 * matches[0] / total if total else None,
            "coverage@0.7": 100 * matches[1] / total if total else None}


def diagnostics(predictions, rows):
    """运行正式评价器并附加短窗口覆盖统计。"""
    result = evaluate(predictions, rows)
    keyed = {p["qid"]: p["pred_relevant_windows"] for p in predictions}
    result["short_gt_coverage"] = short_coverage([keyed[r["qid"]] for r in rows], rows)
    return result


def save_best(path, payload):
    """原子写入可独立重载的短窗口权重和 gate 参数。"""
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(args, backbone, config, layout, device, contract, backbone_epoch):
    """运行可恢复的短窗口训练，并在每个评价点用训练留出集拟合门。"""
    output = args.output_dir
    train_rows, dev_rows = read_jsonl(args.train_annotations), read_jsonl(args.annotations)
    assert_disjoint({"train": train_rows, "dev": dev_rows})
    rows_by_id = {r["qid"]: r for r in train_rows}
    head_ids, gate_ids = split_for_gate(train_rows)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    head = make_head(config["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max(1, int(args.epochs * .75)), gamma=.1)
    history, best = [], None
    log = output / "train_log.jsonl"
    last_path = output / "model_last.pt"
    if args.resume:
        state = restore_training_state(args.resume, head, optimizer, scheduler, generator, contract)
        history, best = state["history"], state["best"]
        write_jsonl(log, history)
        if best is not None:
            save_best(output / "model_best.pt", best)
            dump_json(output / f"dev_metrics_epoch_{best['epoch']}.json", state["best_metrics"])
            write_jsonl(output / f"dev_predictions_epoch_{best['epoch']}.jsonl", state["best_predictions"])
        best_metrics, best_predictions = state["best_metrics"], state["best_predictions"]
        print(json.dumps({"resume_from": str(args.resume), "next_epoch": len(history) + 1}), flush=True)
    elif last_path.exists() or log.exists():
        raise FileExistsError(f"Existing run in {output}. Use --resume or a fresh output directory.")
    else:
        best_metrics, best_predictions = None, None
    # 缓存不消耗 train shuffle 随机流；eval 模式使重建缓存与恢复兼容。
    print(json.dumps({"seed": args.seed, "stage": "cache_frozen_features"}), flush=True)
    cached = cache_split(backbone, config, layout, args.train_annotations, "train", device, args.batch_size)
    by_id = {r.prediction["qid"]: r for r in cached}
    train_set, gate_set = ([by_id[qid] for qid in ids] for ids in (head_ids, gate_ids))
    dev_set = cache_split(backbone, config, layout, args.annotations, "dev", device, args.batch_size)
    if [r.prediction["qid"] for r in dev_set] != [r["qid"] for r in dev_rows]:
        raise ValueError("Dataset dev order differs from annotation order")
    base = [r.prediction for r in dev_set]
    baseline = diagnostics(base, dev_rows)
    dump_json(output / "baseline_metrics.json", baseline)
    write_jsonl(output / "baseline_predictions.jsonl", base)
    dump_json(output / "train_split.json", {"head_qids": head_ids, "gate_qids": gate_ids})
    print(json.dumps({"seed": args.seed, "backbone_epoch": backbone_epoch,
                      "head_queries": len(train_set), "gate_queries": len(gate_set),
                      "dev_queries": len(dev_set), "baseline": baseline["brief"]}), flush=True)

    end_epoch = args.epoch_limit or args.epochs
    if end_epoch < len(history):
        raise ValueError("Epoch limit precedes the restored checkpoint")
    for epoch in range(len(history) + 1, end_epoch + 1):
        head.train()
        order = torch.randperm(len(train_set), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), args.batch_size):
            records = [train_set[i] for i in order[start:start + args.batch_size]]
            logits, windows, valid = score_batch(head, records, device)
            targets = short_targets(windows, [rows_by_id[r.prediction["qid"]] for r in records])
            loss = score_loss(logits, targets, valid)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite short-window loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(records)
        record = {"epoch": epoch, "train_loss": total / len(train_set),
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            gate = calibrate(gate_set, ranked_short(head, gate_set, device, args.batch_size), rows_by_id)
            ranked = ranked_short(head, dev_set, device, args.batch_size)
            predictions, changed = merge(dev_set, ranked, gate)
            always, always_changed = merge(dev_set, ranked, None)
            metrics = diagnostics(predictions, dev_rows)
            if any(p["pred_relevant_windows"][:2] != b["pred_relevant_windows"][:2]
                   for p, b in zip(predictions, base, strict=True)):
                raise RuntimeError("Protected Top-2 changed")
            value = metrics["brief"]["MR-full-mAP"]
            record.update(dev_mAP=value, replacements=changed)
            report = {
                "gated": metrics, "always_replace": diagnostics(always, dev_rows),
                "replacements": changed, "always_replacements": always_changed,
                "short_branch_top1": short_coverage([w[:1] for w in ranked], dev_rows),
                "short_branch_top3": short_coverage(ranked, dev_rows),
            }
            dump_json(output / f"dev_metrics_epoch_{epoch}.json", report)
            write_jsonl(output / f"dev_predictions_epoch_{epoch}.jsonl", predictions)
            if best is None or value > best["dev_mAP"]:
                best = {"epoch": epoch, "dev_mAP": value, "gate": gate,
                        "head": {k: t.detach().cpu().clone() for k, t in head.state_dict().items()},
                        "backbone_sha256": contract["backbone_sha256"]}
                best_metrics, best_predictions = report, predictions
                save_best(output / "model_best.pt", best)
        history.append(record)
        save_training_state(last_path, head, optimizer, scheduler, generator,
                            {"contract": contract, "history": history, "best": best,
                             "best_metrics": best_metrics, "best_predictions": best_predictions})
        write_jsonl(log, history)
        print(json.dumps(record), flush=True)

    if end_epoch < args.epochs:
        return
    dump_json(output / "summary.json", {
        "seed": args.seed, "best_epoch": best["epoch"], "best_dev_mAP": best["dev_mAP"],
        "baseline_dev_mAP": baseline["brief"]["MR-full-mAP"], "backbone_epoch": backbone_epoch,
        "head_queries": len(head_ids), "gate_queries": len(gate_ids), "contract": contract,
    })


def main():
    """从课程数据运行训练或冻结权重推理。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path)
    parser.add_argument("--inference-checkpoint", type=Path)
    parser.add_argument("--split", choices=("dev", "final"), default="dev")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epoch-limit", type=int, help="Pause after this epoch with the same training contract")
    parser.add_argument("--seed", type=int, choices=(42, 2026, 3407), default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_every, args.lr) <= 0:
        parser.error("Positive training settings required")
    if args.epoch_limit is not None and not 1 <= args.epoch_limit <= args.epochs:
        parser.error("--epoch-limit must fall within the original training budget")
    if bool(args.inference_checkpoint) == bool(args.train_annotations):
        parser.error("Supply either --train-annotations or --inference-checkpoint")
    if args.train_annotations and args.split != "dev":
        parser.error("Training requires dev for selection")
    device = torch.device(args.device)
    layout = discover_features(args.feature_root)
    signature = sha256(args.backbone)
    backbone, config, seed, epoch = load_backbone(args.backbone, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.inference_checkpoint:
        state = torch.load(args.inference_checkpoint, map_location="cpu", weights_only=True)
        if state["backbone_sha256"] != signature:
            raise ValueError("The short-window checkpoint requires its original frozen backbone")
        head = make_head(config["hidden_dim"]).to(device)
        head.load_state_dict(state["head"], strict=True)
        records = cache_split(backbone, config, layout, args.annotations, args.split, device, args.batch_size)
        predictions, _ = merge(records, ranked_short(head, records, device, args.batch_size), state["gate"])
        write_jsonl(args.output_dir / "predictions.jsonl", predictions)
    else:
        if seed != args.seed:
            raise ValueError("Training seed must match frozen backbone seed")
        contract = dict(backbone_sha256=signature, train_sha256=sha256(args.train_annotations),
                        dev_sha256=sha256(args.annotations), feature_layout=layout, seed=seed,
                        device=args.device, epochs=args.epochs, batch_size=args.batch_size,
                        eval_every=args.eval_every, lr=args.lr)
        train(args, backbone, config, layout, device, contract, epoch)


if __name__ == "__main__":
    main()
