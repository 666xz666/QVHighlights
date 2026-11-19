from types import SimpleNamespace

import torch

from locator_common import model_config
from student_code.qmt_model import build_qmt_model
from student_code import train_qmt_locator


def qmt_config(**overrides):
    config = model_config("cpu")
    config.update(
        qmt_cross_attn=True, qmt_multiscale=True, qmt_hard_negative=True,
        qmt_num_heads=8, qmt_dilations=(1, 2, 4), qmt_gate_bias=-2.0,
        qmt_hard_negative_coef=0.1, qmt_hard_negative_margin=0.2,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def test_qmt_configuration_is_checkpoint_reloadable():
    torch.manual_seed(7)
    model, _ = build_qmt_model(qmt_config())
    model.eval()
    video = torch.randn(2, 5, model.transformer.d_model)
    text = torch.randn(2, 3, model.transformer.d_model)
    video_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    expected = model._refine_video(video, text, text_mask, video_mask)

    restored, _ = build_qmt_model(qmt_config())
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.eval()
    actual = restored._refine_video(video, text, text_mask, video_mask)
    torch.testing.assert_close(actual, expected)

    raw_video = torch.randn(2, 5, qmt_config().v_feat_dim)
    raw_text = torch.randn(2, 3, qmt_config().t_feat_dim)
    with torch.inference_mode():
        expected_predictions = model(raw_text, text_mask, raw_video, video_mask)
        actual_predictions = restored(raw_text, text_mask, raw_video, video_mask)
    for key in ("pred_logits", "pred_spans", "saliency_scores"):
        torch.testing.assert_close(actual_predictions[key], expected_predictions[key])


def test_qmt_parameters_receive_updates():
    torch.manual_seed(11)
    model, _ = build_qmt_model(qmt_config())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = model.qmt_gate.weight.detach().clone()
    refined = model._refine_video(
        torch.randn(2, 5, model.transformer.d_model),
        torch.randn(2, 3, model.transformer.d_model),
        torch.ones(2, 3, dtype=torch.long),
        torch.ones(2, 5, dtype=torch.long),
    )
    refined.square().mean().backward()
    optimizer.step()
    assert not torch.equal(model.qmt_gate.weight.detach(), before)


def test_hard_negative_margin_uses_gt_interior_and_exterior():
    _, criterion = build_qmt_model(qmt_config())
    outputs = {
        "saliency_scores": torch.tensor([[0.0, 1.0, 2.0, 4.0]]),
        "video_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    targets = {"span_labels": [{"spans": torch.tensor([[0.375, 0.25]])}]}
    loss = criterion.loss_qmt_hard_negative(outputs, targets)
    assert loss.item() > 0


def test_qmt_dev_evaluation_uses_row_contract_and_brief_map():
    predictions = [{"qid": 1}]
    dev_rows = [{"qid": 1}]
    calls = []
    original_evaluate = train_qmt_locator.evaluate

    def fake_evaluate(actual_predictions, actual_dev_rows):
        calls.append((actual_predictions, actual_dev_rows))
        return {"brief": {"MR-full-mAP": 12.34}}

    try:
        train_qmt_locator.evaluate = fake_evaluate
        metrics, current_map = train_qmt_locator.evaluate_dev_predictions(predictions, dev_rows)
    finally:
        train_qmt_locator.evaluate = original_evaluate

    assert calls == [(predictions, dev_rows)]
    assert metrics["brief"]["MR-full-mAP"] == 12.34
    assert current_map == 12.34
