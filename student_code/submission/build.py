"""调用原版 student-kit 打包，并在 ZIP 外保存可追溯的权重清单。"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path


def digest(path):
    """流式计算文件 SHA256，避免一次加载整个模型。"""
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    """校验显式三种子输入；不修改原始训练产物，也不自动选最佳方法。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--kind', choices=['a', 'c'], required=True)
    p.add_argument('--mode', choices=['pointwise', 'pairwise'])
    p.add_argument('--output', type=Path, required=True)
    for seed in (42, 2026, 3407):
        p.add_argument(f'--seed{seed}', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or Path(str(a.output) + '.manifest.json').exists():
        p.error('Output already exists; choose a new filename')
    sys.path.insert(0, str(a.kit.resolve()))
    from evalkit.package import package
    paths = [getattr(a, f'seed{s}').resolve() for s in (42, 2026, 3407)]
    records, configs = [], []
    for seed, path in zip((42, 2026, 3407), paths):
        record = {'seed': seed, 'source': str(path)}
        if a.kind == 'a':
            import torch
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            if ckpt.get('seed') != seed or ckpt.get('schema_version') != 2:
                raise ValueError(f'Seed/schema mismatch: {path}')
            if ckpt.get('model_family') != 'qmt_detr':
                raise ValueError('Use the official packager for baseline A0')
            configs.append(ckpt['model_config'])
            record.update(epoch=ckpt['epoch'], metrics=ckpt['metrics'],
                          model_config=ckpt['model_config'])
            files = [path]
            del ckpt
        else:
            if not a.mode:
                p.error('C requires --mode')
            files = [path / n for n in ('adapter_config.json', 'adapter_model.safetensors')]
            configs.append(json.loads(files[0].read_text()))
        record['files'] = {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size} for f in files}
        records.append(record)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('Three seeds have different model/adapter configurations')
    with tempfile.TemporaryDirectory() as tmp:
        custom = None
        if a.kind == 'a':
            custom = Path(tmp)
            here = Path(__file__).resolve().parent
            shutil.copy2(here / 'infer.py', custom / 'infer.py')
            shutil.copy2(here.parent / 'qmt/model.py', custom / 'qmt_model.py')
        package(a.kind, a.output, paths, mode=a.mode, custom=custom)
    manifest = {'kind': a.kind, 'mode': a.mode, 'seeds': records,
                'zip_sha256': digest(a.output), 'zip_bytes': a.output.stat().st_size,
                'kit_sha256': {n: digest(a.kit / 'evalkit' / n)
                               for n in ('package.py', 'submission.py')}}
    Path(str(a.output) + '.manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
