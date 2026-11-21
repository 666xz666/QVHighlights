"""验证课程两路门控的恒等初始化、掩码、更新和 checkpoint 契约。"""
import io

import torch

from student_code.tests.test_qmt_model import qmt_config
from student_code.qmt.model import build_qmt_model


def test_feature_gate_contract():
    """在完整模型维度上验证门控开关、梯度与保存后预测一致性。"""
    for cross in (False, True):
        config = qmt_config(qmt_cross_attn=cross, qmt_multiscale=False,
                            qmt_hard_negative=False, qmt_feature_gate=False)
        torch.manual_seed(42)
        baseline, _ = build_qmt_model(config)
        baseline_rng = torch.get_rng_state()
        config.qmt_feature_gate = True
        torch.manual_seed(42)
        model, _ = build_qmt_model(config)
        assert torch.equal(baseline_rng, torch.get_rng_state())
        baseline.eval()
        model.eval()
        video = torch.randn(2, 5, 2818)
        text = torch.randn(2, 3, 512)
        tm = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        vm = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
        gates = model.feature_gates(text, tm)
        assert torch.equal(gates, torch.ones_like(gates))
        assert torch.equal(model._gate_features(video, gates), video)
        with torch.no_grad():
            original = baseline(text, tm, video, vm)
            actual = model(text, tm, video, vm)
        for key in ("pred_logits", "pred_spans", "saliency_scores"):
            torch.testing.assert_close(original[key], actual[key], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.qmt_feature_gate_mlp[-1].weight.detach().clone()
        out = model(text, tm, video, vm)
        (out["pred_logits"].square().mean() + out["pred_spans"].mean()).backward()
        assert model.qmt_feature_gate_mlp[-1].weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, model.qmt_feature_gate_mlp[-1].weight)
        altered = text.clone()
        altered[~tm] = 1000
        gates = model.feature_gates(text, tm)
        torch.testing.assert_close(gates, model.feature_gates(altered, tm), rtol=0, atol=0)
        assert not torch.equal(gates[0], gates[1])
        assert torch.equal(model._gate_features(video, gates)[..., -2:], video[..., -2:])
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored, _ = build_qmt_model(config)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        restored.eval()
        with torch.no_grad():
            expected = model(text, tm, video, vm)
            actual = restored(text, tm, video, vm)
        for key in ("pred_logits", "pred_spans", "feature_gates"):
            torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
