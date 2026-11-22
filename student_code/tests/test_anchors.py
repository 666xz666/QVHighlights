"""真实两样本验证动态锚点、梯度、padding、导出及优化器恢复一致性。"""
import random
import tempfile
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'upstream/moment_detr')]
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from locator_common import model_config, discover_features, make_dataset, predict
from moment_detr.start_end_dataset import start_end_collate, prepare_batch_inputs
from student_code.qmt.model import build_qmt_model
from student_code.qmt.resume import save_training_state, restore_training_state
from student_code.qmt.train import build_lr_scheduler


def check_mode(mode):
    """全维度真实数据两步优化，恢复后下一步与Top-3导出必须一致。

    Args:
        mode (str): uniform或multiscale。

    Returns:
        None: 断言失败即停止，不写正式checkpoint。
    """
    config = model_config('cpu')
    config.update(qmt_cross_attn=True, qmt_tcn=True, qmt_anchor_mode=mode,
                  qmt_multiscale=False, qmt_hard_negative=False)
    torch.manual_seed(42)
    model, criterion = build_qmt_model(SimpleNamespace(**config))
    dataset = make_dataset(str(ROOT / 'data/course/train.jsonl'),
                           discover_features(ROOT / 'data/features'), config, load_labels=True)
    batch = start_end_collate([dataset[0], dataset[1]])
    inputs, targets = prepare_batch_inputs(batch[1], 'cpu')
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
        expected = model.qmt_anchors.logits.sigmoid()[None].expand(2, -1, -1)
        torch.testing.assert_close(out['pred_spans'], expected)
        assert out['pred_spans'].shape == (2, 10, 2)
        assert len(out['aux_outputs']) == 1
        torch.testing.assert_close(out['aux_outputs'][0]['pred_spans'], expected)
        # 人工追加padding，确保真实样本长度不变且padding不污染有效预测。
        padded = {k: v.clone() for k, v in inputs.items()}
        padded['src_vid'] = torch.cat((inputs['src_vid'], torch.randn(2, 3, 2818)), 1)
        padded['src_vid_mask'] = torch.cat((inputs['src_vid_mask'],
                                          inputs['src_vid_mask'].new_zeros(2, 3)), 1)
        other = model(**padded)
        torch.testing.assert_close(out['pred_logits'], other['pred_logits'], atol=2e-6, rtol=2e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    scheduler = build_lr_scheduler(optimizer, 90)
    generator = torch.Generator().manual_seed(42)

    def step(net, opt):
        net.train()
        losses = criterion(net(**inputs), targets)
        loss = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for param in (net.qmt_anchors.logits, net.qmt_anchors.position[-1].weight,
                      net.span_embed.layers[-1].weight):
            assert param.grad is not None and param.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(net.parameters(), .1)
        opt.step()
        return float(loss.detach())

    first = step(model, optimizer)
    scheduler.step()
    restored, _ = build_qmt_model(SimpleNamespace(**config))
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=1.5e-4)
    restored_scheduler = build_lr_scheduler(restored_opt, 90)
    restored_generator = torch.Generator()
    with tempfile.TemporaryDirectory(prefix='qvh_anchor_smoke_') as directory:
        path = Path(directory) / 'model_last.pt'
        save_training_state(path, model, optimizer, scheduler, generator,
                            {'contract': config, 'epoch': 1, 'history': [{'loss': first}]})
        expected_random = (random.random(), float(np.random.rand()),
                           torch.randperm(10, generator=generator))
        a = step(model, optimizer)
        scheduler.step()
        metadata = restore_training_state(path, restored, restored_opt,
                                          restored_scheduler, restored_generator, config)
        assert metadata['epoch'] == 1 and metadata['history'][0]['loss'] == first
        assert random.random() == expected_random[0]
        assert float(np.random.rand()) == expected_random[1]
        assert torch.equal(torch.randperm(10, generator=restored_generator), expected_random[2])
        b = step(restored, restored_opt)
        restored_scheduler.step()
        assert scheduler.state_dict() == restored_scheduler.state_dict()
        try:
            restore_training_state(path, restored, restored_opt, restored_scheduler,
                                   restored_generator, {'wrong': True})
        except ValueError:
            pass
        else:
            raise AssertionError('Changed resume contract was accepted')
    assert a == b
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, collate_fn=start_end_collate)
    pa = predict(model, loader, torch.device('cpu'), 'train')
    pb = predict(restored, loader, torch.device('cpu'), 'train')
    assert pa == pb
    assert all(len(row['pred_relevant_windows']) == 3 for row in pa)
    print(mode, 'step1', first, 'restored_step2', a, 'Top3_equal', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    for mode in (sys.argv[1:] or ('uniform', 'multiscale')):
        check_mode(mode)
    print('anchor checks passed', flush=True)
