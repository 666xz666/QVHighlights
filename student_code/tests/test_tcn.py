"""验证串联时序卷积与匹配对照的初始化、掩码和重载契约。"""
import io
import torch
from student_code.tests.test_qmt_model import qmt_config
from student_code.qmt.model import build_qmt_model


def test_tcn_contract():
    """完整模型维度上验证两种对照、参数更新和padding不回流。"""
    for cross, gate in ((False, False), (True, False), (False, True)):
        config = qmt_config(qmt_cross_attn=cross, qmt_multiscale=False,
                            qmt_hard_negative=False, qmt_tcn=False, qmt_feature_gate=gate)
        torch.manual_seed(42)
        baseline, _ = build_qmt_model(config)
        rng = torch.get_rng_state()
        config.qmt_tcn = True
        torch.manual_seed(42)
        model, _ = build_qmt_model(config)
        assert torch.equal(rng, torch.get_rng_state())
        model.eval()
        baseline.eval()
        video, text = torch.randn(2, 7, 2818), torch.randn(2, 3, 512)
        vm = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1]*7], dtype=torch.bool)
        tm = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            expected = baseline(text, tm, video, vm)
            actual = model(text, tm, video, vm)
        for key in ('pred_logits', 'pred_spans', 'saliency_scores'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.qmt_tcn_layers[-1].weight.detach().clone()
        actual = model(text, tm, video, vm)
        actual['pred_logits'].square().mean().backward()
        assert model.qmt_tcn_layers[-1].weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, model.qmt_tcn_layers[-1].weight)
        projected = torch.randn(2, 7, 256, requires_grad=True)
        modified = projected.detach().clone()
        modified[~vm] = 1000
        output = model._temporal_residual(projected, vm)
        torch.testing.assert_close(output[vm], model._temporal_residual(modified, vm)[vm])
        output[vm].sum().backward()
        assert torch.count_nonzero(projected.grad[~vm]) == 0
        assert model.qmt_tcn_layers[0].weight.grad.abs().sum() > 0
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored, _ = build_qmt_model(config)
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        restored.eval()
        with torch.no_grad():
            expected = model(text, tm, video, vm)
            actual = restored(text, tm, video, vm)
        for key in ('pred_logits', 'pred_spans', 'saliency_scores'):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
