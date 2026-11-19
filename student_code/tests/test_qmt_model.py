from types import SimpleNamespace

import torch

from locator_common import model_config
from student_code.qmt.model import build_qmt_model
from student_code.qmt import train as train_qmt_locator


def qmt_config(**overrides):
    config = model_config("cpu")
    config.update(
        qmt_cross_attn=True, qmt_multiscale=True, qmt_hard_negative=True,
        qmt_num_heads=8, qmt_dilations=(1, 2, 4), qmt_gate_bias=-2.0,
        qmt_hard_negative_coef=0.1, qmt_hard_negative_margin=0.2,
        qmt_negative_query=False, qmt_negative_query_coef=0.1,
        qmt_negative_query_margin=0.2,
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


def test_negative_query_margin_uses_all_valid_mismatched_clips():
    _, criterion = build_qmt_model(qmt_config(qmt_negative_query=True))
    positive_scores = torch.tensor(
        [[0.0, 2.0, 4.0, 0.0]], requires_grad=True,
    )
    outputs = {
        "saliency_scores": positive_scores,
        "video_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.bool),
    }
    targets = {"saliency_pos_labels": torch.tensor([[1, 2]])}
    easy_mismatch = {"saliency_scores": torch.tensor([[0.0, 0.0, 0.0, 99.0]])}
    negative_scores = torch.tensor(
        [[0.0, 3.5, 0.0, 99.0]], requires_grad=True,
    )
    hard_mismatch = {"saliency_scores": negative_scores}
    easy_loss = criterion.loss_qmt_negative_query(outputs, easy_mismatch, targets)
    hard_loss = criterion.loss_qmt_negative_query(outputs, hard_mismatch, targets)
    assert easy_loss.item() == 0.0
    assert hard_loss.item() > 0.0
    hard_loss.backward()
    assert positive_scores.grad[0, 1].item() < 0.0
    assert positive_scores.grad[0, 2].item() < 0.0
    assert negative_scores.grad[0, 1].item() > 0.0
    assert negative_scores.grad[0, 3].item() == 0.0


def test_cyclic_mismatch_rotates_text_only_and_skips_singletons():
    model_inputs = {
        "src_vid": torch.tensor([[10.0], [20.0], [30.0]]),
        "src_vid_mask": torch.ones(3, 1),
        "src_txt": torch.tensor([[1.0], [2.0], [3.0]]),
        "src_txt_mask": torch.tensor([[1], [1], [0]]),
    }
    mismatched = train_qmt_locator.make_cyclic_mismatch(model_inputs)
    assert mismatched is not None
    assert mismatched["src_vid"] is model_inputs["src_vid"]
    torch.testing.assert_close(mismatched["src_txt"], torch.tensor([[3.0], [1.0], [2.0]]))
    torch.testing.assert_close(mismatched["src_txt_mask"], torch.tensor([[0], [1], [1]]))
    singleton = {key: value[:1] for key, value in model_inputs.items()}
    assert train_qmt_locator.make_cyclic_mismatch(singleton) is None


def test_trcc_zero_init_is_identity_and_checkpoint_reloadable():
    model, _ = build_qmt_model(qmt_config(
        qmt_multiscale=False, qmt_hard_negative=False, qmt_trcc=True,
        qmt_trcc_temperature=0.02, qmt_trcc_context_scale=2.0,
    ))
    logits = torch.randn(2, 3, 2)
    spans = torch.rand(2, 3, 2)
    saliency = torch.randn(2, 5)
    video_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    actual = model._calibrate_logits(logits, spans, saliency, video_mask)
    torch.testing.assert_close(actual, logits, rtol=0.0, atol=0.0)

    restored, _ = build_qmt_model(qmt_config(
        qmt_multiscale=False, qmt_hard_negative=False, qmt_trcc=True,
        qmt_trcc_temperature=0.02, qmt_trcc_context_scale=2.0,
    ))
    model.qmt_trcc_alpha.data.fill_(0.4)
    model.qmt_trcc_beta.data.fill_(0.2)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.qmt_trcc_alpha.item() == model.qmt_trcc_alpha.item()
    assert restored.qmt_trcc_beta.item() == model.qmt_trcc_beta.item()

    model.eval()
    restored.eval()
    config = qmt_config()
    raw_video = torch.randn(2, 5, config.v_feat_dim)
    raw_text = torch.randn(2, 3, config.t_feat_dim)
    text_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.inference_mode():
        expected = model(raw_text, text_mask, raw_video, video_mask)
        actual = restored(raw_text, text_mask, raw_video, video_mask)
    for key in ("pred_logits", "pred_spans", "saliency_scores"):
        torch.testing.assert_close(actual[key], expected[key])


def test_trcc_ignores_padded_saliency():
    model, _ = build_qmt_model(qmt_config(
        qmt_multiscale=False, qmt_hard_negative=False, qmt_trcc=True,
    ))
    model.qmt_trcc_alpha.data.fill_(0.7)
    model.qmt_trcc_beta.data.fill_(0.3)
    logits = torch.zeros(1, 1, 2)
    spans = torch.tensor([[[0.5, 0.5]]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    base = torch.tensor([[0.0, 2.0, 2.0, 0.0, -10.0, -10.0]])
    changed_padding = base.clone()
    changed_padding[:, 4:] = 1e4
    expected = model._calibrate_logits(logits, spans, base, mask)
    actual = model._calibrate_logits(logits, spans, changed_padding, mask)
    torch.testing.assert_close(actual, expected)


def test_trcc_gradient_updates_calibration_without_span_gradient():
    model, _ = build_qmt_model(qmt_config(
        qmt_multiscale=False, qmt_hard_negative=False, qmt_trcc=True,
    ))
    logits = torch.zeros(1, 1, 2)
    spans = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
    saliency = torch.tensor([[0.0, 2.0, 2.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    calibrated = model._calibrate_logits(logits, spans, saliency, mask)
    (-calibrated[..., 0].mean()).backward()
    assert model.qmt_trcc_alpha.grad.item() < 0.0
    assert model.qmt_trcc_beta.grad.item() < 0.0
    assert spans.grad is None


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
