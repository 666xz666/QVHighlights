# 串联时序卷积实验

对应课程文档的轻量残差时序卷积，不是此前并行depthwise多尺度分支。
投影后使用两层普通Conv1d，通道256、核3、dilation为1和2、对称padding，
中间ReLU，串联感受野7个clip。卷积前及每层后清除padding。
末层零初始化，隔离构造随机数，使初始预测与关闭分支的对照一致。
新增参数393728个；不改定位损失、候选导出或评分器。

实验固定60轮、batch32、AdamW学习率1e-4、StepLR第45轮衰减、每5轮评价，
种子42/2026/3407。每种子按dev最佳epoch选模，再报告均值、样本标准差及配对差值。

|实验|匹配对照|额外模块|
|---|---|---|
|tcn-only|A0_epoch60|串联卷积|
|cross-tcn|QMT_epoch60/cross|Cross后串联卷积|

两组关闭多尺度、困难负样本、负查询、TRCC、Bridge、IoU排序和特征门控。
复用同预算对照，至少2/3种子改善且平均提升0.30个百分点作为接受标准。

```bash
export QVH_PYTHON=/data1/xz/miniconda3/envs/QVHighlights/bin/python
bash student_code/scripts/run_qmt_queue.sh 0 tcn-only --epochs 60 42 2026 3407
bash student_code/scripts/run_qmt_queue.sh 0 cross-tcn --epochs 60 42 2026 3407
```

队列跳过完成种子；未完成种子不支持优化器中途恢复，重跑前需保留并移出其旧输出。
