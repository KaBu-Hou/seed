# EEG- and Micro-Expression-Based Emotion Recognition and Consciousness Detection 复现报告

## 当前结论

我已完成第一轮论文阅读、数据结构扫描，并搭建了可运行的复现代码框架。当前本地数据是 `SEED/SJTU` 公开数据：包含 15 名被试、3 次会话、每次 15 个 trial 的 EEG 预处理数据与 1 秒级 EEG 特征，也包含情绪诱发影片 `Stimuli`。本地没有发现被试面部反应视频或作者自采 DoC 患者数据，因此完整的“EEG + 微表情 + 意识检测”只能做成可插拔框架；可忠实启动的部分是论文 Table III 的 SEED EEG 三分类复现。

## 本地目录结构

完整扫描结果已保存到：

- `processed/seed_directory_tree.txt`
- `processed/seed_summary.json`

压缩结构如下：

```text
SEED/SJTU/
  ExtractedFeatures_1s/
    45 个被试会话 .mat 文件
    label.mat
    readme.txt
  Preprocessed_EEG/
    45 个被试会话 .mat 文件
    label.mat
    readme.txt
  Stimuli/
    negative/ 5 个负性诱发视频
    neutral/ 5 个中性诱发视频
    positive/ 5 个正性诱发视频
  Data description.docx
  被试.doc
```

数据说明确认：SEED 标签顺序为 `1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1`，其中 `1=positive`，`0=neutral`，`-1=negative`。`ExtractedFeatures_1s` 中 `de_LDS1` 示例形状为 `(62, 235, 5)`，含 62 EEG 通道、235 个 1 秒片段、5 个频段。

## 论文方法要点

论文任务包含三分类情绪识别与 DoC 意识检测。情绪类别为 positive、neutral、negative。意识检测不是直接二分类训练，而是先得到患者情绪概率，再把情绪概率映射到连续情绪分数，并与健康被试情绪向量计算欧氏距离和余弦相似度；距离越小、相似度越高，表示残余意识/情绪响应越接近健康基线。

EEG 特征使用 Differential Entropy，按 1 秒非重叠窗提取，频段为 `delta 0.1-4 Hz`、`theta 4-8 Hz`、`alpha 8-12 Hz`、`beta 12-30 Hz`、`gamma 30-45 Hz`。公开 SEED 数据论文描述为 62 通道，1000 Hz 降采样到 200 Hz，三分类。

微表情分支使用 MTCNN 做人脸检测与对齐，从约 0.2 秒内均匀采样 5 帧，灰度化、直方图均衡、标准化并 resize 到 `56 x 56`，形成 `5 x 56 x 56` 张量。

融合策略是 modality-respecting STAE：EEG 使用空间注意力，微表情使用时间注意力，之后通过门控残差的 domain-coordinated 机制融合，再送入 STST/Swin Transformer 与 MLP Head。训练设置为 PyTorch、AdamW、学习率 `3e-4`、cosine annealing、batch size `32`、`100` epochs；公开数据采用 LOSO-CV。指标包括 Accuracy、UF1、UAR；DoC 评估使用 Euclidean distance 和 cosine similarity。

论文中 SEED 对比目标为：

| Protocol | Paper STST Acc | Std |
|---|---:|---:|
| Subject-dependent | 93.47% | 5.47% |
| Subject-independent | 86.43% | 7.27% |

## 采用的技术路线

1. 以 `ExtractedFeatures_1s/de_LDS` 为主输入，避免重复滤波和 DE 计算，直接构建 `(samples, 62, 5)` EEG 张量。
2. 标签映射为 `negative=0`、`neutral=1`、`positive=2`。
3. 严格按 subject 做 LOSO subject-independent 评估；脚本也保留调试子集参数。
4. 模型实现保留论文核心：EEG 空间注意力、微表情时间注意力、门控融合、MLP 分类头。由于本地没有 `timm`，当前用轻量 CNN/MLP 头替代完整 Swin Transformer，并在代码注释中标明该假设。
5. 微表情脚本支持从视频抽取 `5 x 56 x 56` 张量。当前本地只有刺激影片，不是被试面部录像，所以默认产物命名为 `me_stimulus_proxy`，仅作为接口与替代特征验证，不能当作真实微表情结果。
6. DoC 意识检测部分将在缺少作者自采 21 名患者数据时保留指标计算接口和报告说明，不能声称复现 Table VIII 的患者结论。

## 已生成代码

- `code/explore_data.py`：扫描数据结构，输出目录树和数据摘要。
- `code/preprocess_eeg.py`：读取 SEED 1 秒 DE/PSD 等特征，生成 NumPy 数据集。
- `code/preprocess_me.py`：从视频抽取 5 帧 56x56 张量；当前用于刺激视频代理。
- `code/model.py`：STAE/STST 风格 PyTorch 模型。
- `code/train_eval.py`：LOSO 训练、Accuracy/Macro-F1/UAR、混淆矩阵和曲线输出。
- `code/utils.py`：随机种子、路径、标签、指标、绘图工具。
- `requirements.txt`：当前依赖列表。

## 运行命令

```powershell
python code\explore_data.py
python code\preprocess_eeg.py --out processed\eeg_de_lds.npz
python code\preprocess_me.py --out processed\me_stimulus_proxy.npz
python code\train_eval.py --epochs 100 --batch_size 32 --out_dir outputs\seed_loso
```

快速烟测命令：

```powershell
python code\train_eval.py --max_sessions 4 --subjects 1 2 --epochs 1 --batch_size 128 --hidden 16 --out_dir outputs\smoke
```

## 烟测结果

已完成：

- 数据探索脚本成功，生成 `processed/seed_directory_tree.txt` 和 `processed/seed_summary.json`。
- EEG 预处理小样本成功：`processed/eeg_de_lds_smoke.npz`，形状 `x=(3394, 62, 5)`。
- 微表情代理抽帧成功：`processed/me_stimulus_proxy_smoke.npz`，形状 `x=(15, 5, 56, 56)`。RMVB 文件解码时 OpenCV/FFmpeg 有警告，但仍生成了 15 个张量。
- 模型前向成功：输入 `(4, 62, 5)` 输出 `(4, 3)`。
- 2 个被试、1 epoch 小规模 LOSO 跑通：mean accuracy `0.4764`，mean macro-F1 `0.3688`。这是流程烟测，不是正式结果。

## 当前限制

论文的完整多模态与意识检测结果依赖作者自采数据：10 名健康被试、21 名 DoC 患者、32 通道 EEG 与被试面部视频。本地 SEED 目录没有这部分数据，因此当前不能严格复现患者意识检测 Table VIII。下一步正式实验应先跑完整 SEED EEG LOSO，再把结果与论文 Table III 的 subject-independent STST 结果对比。

## STST/Swin 与 image56 更新

已新增 `--model stst_swin` 和 `--eeg_representation image56`。该路径按论文的 `56 x 56 x n` Swin 输入思想，把 SEED trial 内连续 56 个 1 秒 DE 时间窗重构为 EEG 特征图：

```text
62 x 56 x 5 -> bilinear resize -> 56 x 56 x 5 -> PyTorch B x 5 x 56 x 56
```

这不是从 62 个物理通道中选 56 个通道。论文没有说明 62 通道如何变成 `56 x 56 x 5`，所以这里将其标注为实现假设；`patch_size=2` 来自论文图中 `56 x 56 x n -> 28 x 28 x 4n`，`window_size=7`、`depths=[2,2,2]`、`heads=[2,4,8]`、`mlp_ratio=4.0`、`dropout=0.1` 是 Swin 风格实现假设。

调试命令：

```powershell
python code\train_eval.py --data_npz processed\eeg_de_lds.npz --model stst_swin --eeg_representation image56 --eeg_context 56 --subjects 1 2 --epochs 1 --batch_size 16 --hidden 32 --max_samples_per_subject 200 --out_dir outputs\debug_stst_swin_image56
```

正式 subject-independent 命令：

```powershell
python code\train_eval.py --data_npz processed\eeg_de_lds.npz --model stst_swin --eeg_representation image56 --eeg_context 56 --epochs 100 --batch_size 32 --hidden 64 --out_dir outputs\seed_subject_independent_stst_swin_image56
```
