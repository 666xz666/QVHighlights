# 模型提交

模型 ZIP 与实验记录分开保存。使用教师原版 student-kit，不修改其校验规则。

## A 自研定位器

在仓库根目录执行，三个输入必须属于同一方法和训练预算：

```bash
python student_code/submission/build.py --kit /path/to/student-kit --kind a --output /tmp/A_gate.zip \
  --seed42 outputs/QMT_epoch60/gate-only/seed42/model_best.pt \
  --seed2026 outputs/QMT_epoch60/gate-only/seed2026/model_best.pt \
  --seed3407 outputs/QMT_epoch60/gate-only/seed3407/model_best.pt
```

该命令是用法示例，不代表冻结最终方法。工具校验种子、schema 和模型配置；训练预算需依据各运行 summary 核对。ZIP 只附带 infer.py 和模型模块，不携带训练代码、数据和评分真值。ZIP 外的 manifest 记录逐种子 epoch、指标、配置、文件哈希和打包工具哈希。已有输出不会被覆盖。A0 用官方工具直接打包，不使用自研入口。

## C

使用同一 build.py，指定 --kind c --mode pointwise（或 pairwise），三个 seed 参数指向适配器目录。仅打包 adapter_config.json 和 adapter_model.safetensors。适用于沿用标准推理规则的训练改进；改变推理规则的方法需另写入口。共同 step、训练种子和模式必须在打包前对照训练记录确认，适配器文件本身不足以证明这些信息。

## 验收

先运行官方 validate_archive，再解压到独立目录。以课程目录为 cwd，调用解压所得 custom/infer.py，传入真实 --kind a --seed --device --queries --feature-root --split --checkpoint --output 参数。至少用公开小样本分别验证三个种子，并与仓库导出器输出逐行比对。只通过 ZIP 结构检查不等于推理成功。

服务器代码和输入只读、无外网，输出只能写入临时目录。本入口不依赖开发机路径或训练结果目录，所有模型配置从 checkpoint 恢复。实际容器资源限制仍以平台为准。
