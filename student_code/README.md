# QMT-DETR 与 ILR-QLoRA

本目录包含全部学生方法代码；课程基线、评测器、数据划分和固定候选池均保持不变。

## 方法与协议

QMT-DETR 在 Moment-DETR 投影层之后、视频文本拼接之前加入：

1. 视频作 Query、文本作 Key/Value 的交叉注意力；
2. 负偏置初始化的门控残差；
3. 空洞率为 1、2、4 的深度可分离时序卷积，由文本表示预测尺度权重；
4. 仅作用于最终层的 GT 内部—困难背景 saliency margin loss。

后续受控变体 `cross-neg-query` 保留第 1、2 项并移除多尺度与同视频困难背景损失。训练时将文本沿 batch 维循环移位，构造视频—查询错配，并约束正确配对的标注显著 clip 均值至少比错配查询下所有有效 clip 的最高分高 0.2。该项权重为 0.1；batch size 为 1 时自动跳过。推理路径与候选导出接口不变。

模型配置随 checkpoint 保存，`export_qmt_candidates.py` 只接受 QMT checkpoint，并以 `strict=True` 重载参数。正式 A 实验沿用 40 epoch、batch 32、学习率 `1e-4`、每 5 epoch 评估以及种子 42/2026/3407。

ILR-QLoRA 读取固定的 `data/reference_full/sft_pairwise.jsonl`。每个固定二候选对分别进行 pointwise 前向，使用第一个监督答案 token 之前的
`logit(Yes)-logit(No)` 作为可微分分数。损失为：

```text
L = mean(L_CE) + 0.5 L_listwise + 0.2 L_pairwise
```

`L_listwise` 以候选 IoU 的温度 softmax 为目标分布；`L_pairwise` 以 IoU 差作为 RankNet 权重。每次更新累积 4 对，即严格曝光 8 个候选；936 步共曝光 7,488 个候选。推理继续调用课程 `vlm.rerank`，固定 Top-3 边界不变。

## 快速检查

```bash
cd /data14T/xz/repos/QVHighlights
PYTHONPATH=$PWD:$PWD/scripts:$PWD/upstream/moment_detr \
  /data14T/xz/miniconda3/envs/QVHighlights/bin/python \
  student_code/tests/run_quick_checks.py
```

检查范围包括 ILR 分数位置与梯度方向、C-off 的零辅助损失、QMT 新参数更新，以及 QMT 严格保存/重载后的完整预测一致性。该检查不读取训练数据，也不加载 Qwen 模型。

## A 正式训练

队列脚本的第一个参数是 `CUDA_DEVICE_ORDER=FASTEST_FIRST` 下的 CUDA ordinal，不是 `nvidia-smi` 的物理编号。脚本会在创建或续写实验输出前完成 CUDA 初始化检查，初始化失败时立即退出。同一方法 variant 由非阻塞文件锁保护，重复启动的第二个队列会立即退出，不能并发写同一组输出。

完整方法三种子队列（`<cuda-ordinal>` 替换为已核实且空闲的 ordinal）：

```bash
cd /data14T/xz/repos/QVHighlights
mkdir -p outputs/logs/QMT
nohup bash student_code/run_qmt_queue.sh <cuda-ordinal> full 42 2026 3407 \
  > outputs/logs/QMT/full_queue.out 2>&1 &
tail -f /data14T/xz/repos/QVHighlights/outputs/logs/QMT/full_queue.out
```

逐个种子的 epoch 指标写入 `outputs/logs/QMT/full_seed*.log`。三种子汇总：

```bash
/data14T/xz/miniconda3/envs/QVHighlights/bin/python \
  student_code/summarize_results.py --kind a --root outputs/QMT/full
```

消融队列保持其余预算不变：

```bash
bash student_code/run_qmt_queue.sh <cuda-ordinal> cross 42 2026 3407
bash student_code/run_qmt_queue.sh <cuda-ordinal> cross-multiscale 42 2026 3407
bash student_code/run_qmt_queue.sh <cuda-ordinal> off-multiscale 42 2026 3407
bash student_code/run_qmt_queue.sh <cuda-ordinal> cross-neg-query 42 2026 3407
```

其中 `cross` 仅启用交叉注意力，`cross-multiscale` 再加入多尺度分支，`off-multiscale` 从完整方法中移除多尺度分支但保留困难负样本损失，`cross-neg-query` 在仅 Cross 骨架上加入 Negative-query Saliency Calibration。新校准项默认系数为 `0.1`、margin 为 `0.2`；batch size 小于 2 时自动跳过。

选定配置后导出候选：

```bash
CUDA_DEVICE_ORDER=FASTEST_FIRST CUDA_VISIBLE_DEVICES=<cuda-ordinal> \
  /data14T/xz/miniconda3/envs/QVHighlights/bin/python \
  student_code/export_qmt_candidates.py \
  --checkpoint outputs/QMT/full/seed42/model_best.pt \
  --feature-root data/features --annotations data/course/dev_queries.jsonl \
  --split dev --output outputs/QMT/full/seed42/dev_candidates.jsonl
```

## C 正式训练与评估

完整 ILR 三种子队列会依次训练，并对 312、624、936 三个共同 checkpoint 在固定 dev Top-3 候选池上重排和评分：

```bash
cd /data14T/xz/repos/QVHighlights
mkdir -p outputs/logs/ILR
nohup bash student_code/run_ilr_queue.sh <cuda-ordinal> full 42 2026 3407 \
  > outputs/logs/ILR/full_queue.out 2>&1 &
tail -f /data14T/xz/repos/QVHighlights/outputs/logs/ILR/full_queue.out
```

C-off 保持相同固定对、顺序、累积步数和候选曝光，仅将两个辅助权重设为零：

```bash
nohup bash student_code/run_ilr_queue.sh <cuda-ordinal> off 42 2026 3407 \
  > outputs/logs/ILR/off_queue.out 2>&1 &
tail -f /data14T/xz/repos/QVHighlights/outputs/logs/ILR/off_queue.out
```

共同 step 必须由三种子平均 dev mAP 选择，不得逐种子选择：

```bash
/data14T/xz/miniconda3/envs/QVHighlights/bin/python \
  student_code/summarize_results.py --kind c --root outputs/ILR/full
/data14T/xz/miniconda3/envs/QVHighlights/bin/python \
  student_code/summarize_results.py --kind c --root outputs/ILR/off
```

正式结论报告三种子均值、样本标准差（`ddof=1`）以及相对匹配对照的同种子差值。ILR 的严格主对照为相同步数的 C-off；相对 C0-P 或 C-ZP 的结果可作为整体配置比较，但候选曝光或分组不一致时不解释为单一损失项的因果增益。
