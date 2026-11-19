import torch

from student_code.ilr_objective import (combined_ilr_loss, first_answer_score,
                                        ilr_auxiliary_loss)


def test_first_answer_score_uses_pre_answer_logits_and_keeps_gradient():
    logits = torch.zeros(1, 5, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 2, 3] = 2.5
        logits[0, 2, 4] = -0.5
    labels = torch.tensor([[-100, -100, -100, 3, 2]])
    score = first_answer_score(logits, labels, yes_token_id=3, no_token_id=4)
    assert score.item() == 3.0
    score.backward()
    assert logits.grad[0, 2, 3].item() == 1.0
    assert logits.grad[0, 2, 4].item() == -1.0


def test_ilr_prefers_the_higher_iou_candidate():
    targets = torch.tensor([1.0, 0.0])
    good = torch.tensor([2.0, -1.0], requires_grad=True)
    bad = torch.tensor([-1.0, 2.0], requires_grad=True)
    good_losses = ilr_auxiliary_loss(good, targets)
    bad_losses = ilr_auxiliary_loss(bad, targets)
    assert sum(good_losses) < sum(bad_losses)
    sum(bad_losses).backward()
    assert bad.grad[0] < 0
    assert bad.grad[1] > 0


def test_combined_loss_reports_all_components():
    scores = torch.tensor([0.2, -0.1], requires_grad=True)
    total, components = combined_ilr_loss(
        [torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)],
        scores, torch.tensor([0.7, 0.2]),
    )
    assert set(components) == {"ce", "listwise", "pairwise"}
    assert torch.isfinite(total)
    total.backward()
    assert scores.grad is not None


def test_zero_auxiliary_weights_is_pair_grouped_ce_ablation():
    ce_losses = [torch.tensor(1.0), torch.tensor(3.0)]
    total, _ = combined_ilr_loss(
        ce_losses, torch.tensor([0.2, -0.1]), torch.tensor([1.0, 0.0]),
        listwise_weight=0.0, pairwise_weight=0.0,
    )
    assert total.item() == 2.0
