"""解压 A 提交包，在独立课程目录用少量无标签样本验证三种子推理。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    """隔离 student_code 导入，并与仓库原导出器逐条比较预测。"""
    p = argparse.ArgumentParser()
    p.add_argument('--kit', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--course', type=Path, required=True)
    a = p.parse_args()
    course = a.course.resolve()
    sys.path.insert(0, str(a.kit.resolve()))
    from evalkit.submission import extract_archive
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        extracted = root / 'unpacked'
        extract_archive(a.archive, 'a', extracted)
        isolated = root / 'course'
        isolated.mkdir()
        for name in ('scripts', 'upstream'):
            (isolated / name).symlink_to(course / name, target_is_directory=True)
        queries = root / 'queries.jsonl'
        rows = [json.loads(x) for x in (course / 'data/course/dev_queries.jsonl').read_text().splitlines()[:2]]
        queries.write_text(''.join(json.dumps({k: r[k] for k in ('qid', 'vid', 'query', 'duration')}) + '\n' for r in rows))
        env = dict(os.environ, PYTHONPATH='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        for seed in (42, 2026, 3407):
            checkpoint = extracted / f'seeds/{seed}/locator.pt'
            output, reference = root / f'{seed}.jsonl', root / f'{seed}_reference.jsonl'
            shared = ['--checkpoint', str(checkpoint), '--feature-root', str(course / 'data/features'),
                      '--split', 'dev', '--device', 'cpu']
            subprocess.run([sys.executable, str(extracted / 'custom/infer.py'), '--kind', 'a',
                            '--seed', str(seed), '--queries', str(queries), '--output', str(output),
                            *shared], cwd=isolated, env=env, check=True)
            subprocess.run([sys.executable, str(course / 'student_code/qmt/export_candidates.py'),
                            '--annotations', str(queries), '--output', str(reference), '--workers', '0',
                            *shared], cwd=course, env=env, check=True)
            assert [json.loads(x) for x in output.read_text().splitlines()] == [
                json.loads(x) for x in reference.read_text().splitlines()]
            print(f'Seed {seed}: extracted inference matches repository, 2 queries', flush=True)


if __name__ == '__main__':
    main()
