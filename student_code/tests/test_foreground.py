"""稠密前景监督的梯度、完整标签、初始化及真实数据恢复冒烟测试。"""
import io
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'upstream/moment_detr')]
import torch
from locator_common import model_config, discover_features, make_dataset
from moment_detr.start_end_dataset import start_end_collate, prepare_batch_inputs
from student_code.qmt.model import build_qmt_model
from student_code.qmt.foreground import foreground_losses


def test_masks_and_gradients():
    """完整GT并集、短邻接空隙及padding的梯度方向符合设计。"""
    score = torch.zeros(1, 9, requires_grad=True)
    valid = torch.tensor([[1]*8 + [0]], dtype=torch.bool)
    rows = [{'relevant_windows': [[0, 4], [8, 12]]}]
    dense, gap = foreground_losses(score, valid, rows)
    (dense + gap).backward()
    assert (score.grad[0, [0, 1, 4, 5]] < 0).all()
    assert (score.grad[0, [2, 3, 6, 7]] > 0).all()
    assert score.grad[0, 8] == 0
    altered = score.detach().clone()
    altered[0, 8] = 1000
    for before, after in zip((dense, gap), foreground_losses(altered, valid, rows)):
        torch.testing.assert_close(before, after)
    for windows in ([], [[0, 16]], [[0, 6], [4, 10]]):
        result = foreground_losses(score, valid, [{'relevant_windows': windows}])
        assert all(torch.isfinite(x) for x in result)
        assert result[1] == 0


def test_real_batch_resume():
    """真实维度/数据跑两步，验证初始化等价、backbone梯度和恢复后下一步一致。

    Returns:
        None: 所有断言通过；只在内存保存诊断checkpoint，不触碰正式输出。
    """
    config = model_config('cpu')
    config.update(qmt_cross_attn=True, qmt_tcn=True, qmt_multiscale=False,
                  qmt_hard_negative=False, qmt_dense_foreground=False)
    torch.manual_seed(42)
    baseline, _ = build_qmt_model(SimpleNamespace(**config))
    rng = torch.get_rng_state()
    config['qmt_dense_foreground'] = True
    torch.manual_seed(42)
    model, criterion = build_qmt_model(SimpleNamespace(**config))
    assert torch.equal(rng, torch.get_rng_state())
    dataset = make_dataset(str(ROOT / 'data/course/train.jsonl'),
                           discover_features(ROOT / 'data/features'), config, load_labels=True)
    batch = start_end_collate([dataset[0], dataset[1]])
    inputs, targets = prepare_batch_inputs(batch[1], 'cpu')
    baseline.eval()
    model.eval()
    with torch.no_grad():
        a, b = baseline(**inputs), model(**inputs)
    for key in ('pred_logits', 'pred_spans', 'saliency_scores'):
        assert torch.equal(a[key], b[key])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    import json
    rows = {str(r['qid']): r for r in map(json.loads, (ROOT / 'data/course/train.jsonl').read_text().splitlines())}
    full_rows = [rows[str(r['qid'])] for r in batch[0]]
    model.train()
    out = model(**inputs)
    d, g = foreground_losses(out['foreground_logits'], inputs['src_vid_mask'], full_rows)
    (d + g).backward()
    assert model.qmt_foreground_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.transformer.parameters())
    optimizer.zero_grad(set_to_none=True)

    def step(net, opt):
        net.train()
        output = net(**inputs)
        losses = criterion(output, targets)
        dense, gap = foreground_losses(output['foreground_logits'], inputs['src_vid_mask'], full_rows)
        loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        loss = loss + .1 * dense + .03 * gap
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), .1)
        opt.step()
        return float(loss.detach())

    print('smoke_step1', step(model, optimizer), flush=True)
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'rng': torch.get_rng_state(), 'config': config}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored, _ = build_qmt_model(SimpleNamespace(**saved['config']))
    restored.load_state_dict(saved['model'], strict=True)
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1.5e-4)
    restored_opt.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['rng'])
    first = step(model, optimizer)
    torch.set_rng_state(saved['rng'])
    second = step(restored, restored_opt)
    assert first == second
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    print('smoke_step2_reload_equal', first, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_masks_and_gradients()
    test_real_batch_resume()
    print('foreground checks passed', flush=True)
