"""epoch边界原子保存训练状态；不改变模型、采样或学习率规则。"""
import random

import numpy as np
import torch


def save_training_state(path, model, optimizer, scheduler, generator, metadata):
    """保存模型、优化器、调度器及各随机流，临时文件完成后原子替换。

    Args:
        path (Path): 最近完整epoch的checkpoint路径。
        model (nn.Module): 原始维度模型。
        optimizer (Optimizer): 已完成当前epoch更新的优化器。
        scheduler (LRScheduler): 已完成当前epoch调度的学习率状态。
        generator (Generator): DataLoader专用随机生成器。
        metadata (dict): epoch、最佳模型、日志、数据和配置签名。

    Returns:
        None: 写入一个完整可恢复checkpoint。
    """
    state = dict(metadata=metadata, model=model.state_dict(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 generator=generator.get_state(), python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, scheduler, generator, contract):
    """核对配置及数据签名，然后恢复完整训练状态；不允许静默变更预算。

    Args:
        path (Path): 本地可信训练checkpoint，不接受外部不可信pickle。
        model (nn.Module): 已按原配置构建的模型。
        optimizer (Optimizer): 对应当前模型参数的优化器。
        scheduler (LRScheduler): 对应原训练预算的调度器。
        generator (Generator): DataLoader随机生成器。
        contract (dict): 必须与checkpoint完全相同的配置和数据签名。

    Returns:
        dict: epoch、最佳模型、日志等恢复元数据。
    """
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['metadata']['contract'] != contract:
        raise ValueError('Resume contract differs: data/config/budget must remain identical')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    generator.set_state(state['generator'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('CUDA checkpoint requires CUDA to restore its random stream')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['metadata']
