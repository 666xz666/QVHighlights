"""教师环境 A 推理入口：课程代码来自当前目录，自研模型来自本文件目录。"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    """读取无标签输入并按 checkpoint 配置导出三个候选。"""
    parser = argparse.ArgumentParser()
    for key in ('kind', 'seed', 'device', 'queries', 'feature-root', 'split', 'checkpoint', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    if args.kind != 'a' or int(args.seed) not in (42, 2026, 3407):
        parser.error('This custom entry supports A and the three course seeds only')
    course = Path.cwd()
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(course / 'scripts'),
                   str(course / 'upstream/moment_detr'), str(course)]
    import torch
    from torch.utils.data import DataLoader
    from qmt_model import build_qmt_model
    from locator_common import audit_features, discover_features, make_dataset, predict
    from prepare_course_data import read_jsonl, source_id, write_jsonl
    from evaluate import validate_predictions
    from moment_detr.start_end_dataset import start_end_collate

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('schema_version') != 2 or checkpoint.get('model_family') != 'qmt_detr':
        raise ValueError('Expected QMT schema 2 checkpoint')
    if checkpoint.get('seed') != int(args.seed):
        raise ValueError('Checkpoint seed does not match requested seed')
    rows = read_jsonl(args.queries)
    if args.split not in ('dev', 'final'):
        raise ValueError('Submission supports dev/final only')
    if any(row.get('split', args.split) != args.split for row in rows):
        raise ValueError('Input split mismatch')
    if {row['qid'] for row in rows} & set(checkpoint['train_qids']):
        raise ValueError('Evaluation qids overlap training')
    if {source_id(row['vid']) for row in rows} & set(checkpoint['train_source_ids']):
        raise ValueError('Evaluation videos overlap training')
    config = dict(checkpoint['model_config'], device=args.device)
    layout = discover_features(args.feature_root)
    audit_features(layout, rows)
    model, _ = build_qmt_model(SimpleNamespace(**config))
    model.load_state_dict(checkpoint['model'], strict=True)
    device = torch.device(args.device)
    model.to(device)
    dataset = make_dataset(args.queries, layout, config, load_labels=False)
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False,
                        collate_fn=start_end_collate)
    predictions = predict(model, loader, device, args.split)
    validate_predictions(predictions, rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, predictions)


if __name__ == '__main__':
    main()
