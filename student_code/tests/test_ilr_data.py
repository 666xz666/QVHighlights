from pathlib import Path

from student_code.ilr.train import validate_pairs
from vlm.common import load_jsonl


def test_fixed_pair_pool_matches_ilr_contract():
    path = Path("data/reference_full/sft_pairwise.jsonl")
    rows = load_jsonl(path)
    assert len(rows) == 1248
    validate_pairs(rows)
    assert 936 * 4 * 2 == 7488
