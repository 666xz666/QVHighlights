"""验证固定首窗的边界一致性重排契约。"""

from student_code.qmt.locked_tail_overlap import overlap_support, rerank_tail, temporal_iou


def test_temporal_iou_handles_overlap_and_disjoint_windows():
    assert temporal_iou([0.0, 10.0], [5.0, 15.0]) == 1 / 3
    assert temporal_iou([0.0, 2.0], [2.0, 4.0]) == 0.0


def test_overlap_support_respects_auxiliary_rank():
    auxiliary = [[20.0, 30.0, 3.0], [0.0, 10.0, 2.0]]
    assert overlap_support([20.0, 30.0, 1.0], auxiliary) == 1.0
    assert overlap_support([0.0, 10.0, 1.0], auxiliary) == 0.7


def test_rerank_tail_preserves_first_window_and_swaps_supported_tail():
    primary = [[0.0, 8.0, 9.0], [10.0, 18.0, 8.0], [20.0, 28.0, 7.0]]
    auxiliary = [[20.0, 28.0, 3.0], [30.0, 38.0, 2.0]]
    ranked = rerank_tail(primary, auxiliary, 0.25)
    assert ranked[0] == primary[0]
    assert [window[:2] for window in ranked[1:]] == [[20.0, 28.0], [10.0, 18.0]]
    assert ranked[0][2] > ranked[1][2] > ranked[2][2]
    assert primary[1][2] == 8.0 and primary[2][2] == 7.0


def test_rerank_tail_uses_stable_prior_on_equal_support():
    primary = [[0.0, 8.0, 9.0], [10.0, 18.0, 8.0], [20.0, 28.0, 7.0]]
    auxiliary = [[40.0, 48.0, 3.0]]
    ranked = rerank_tail(primary, auxiliary, float("inf"))
    assert [window[:2] for window in ranked] == [window[:2] for window in primary]
