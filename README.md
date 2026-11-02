# DENS7-PT

**基于 7 个峰密度描述量与随机森林的聚醚三萜（polyether triterpenoid, PT）MS/MS 筛选模型。**

本仓库交付三个已经冻结的 DENS7 随机森林 checkpoint，分别对应三组独立训练数据。三个模型不合并、不共享决策阈值以外的任何参数；其中 **NIST 模型为主模型**，此前内部数据模型作参照，GNPS＋MoNA 模型用于扩展域压力测试。

> 本仓库不包含训练谱图。NIST、GNPS 与 MoNA 数据遵循各自的使用与再分发条款；本仓库只包含代码、特征定义和训练后的模型参数。
>
> 配套阶段报告《NIST与GNPS高难度阴性_峰密度验证与模型交付_阶段报告_20261102.docx》单独提供；本仓库只负责模型与代码交付。

---

## 1. 三个 checkpoint

| checkpoint | 角色 | 训练谱图 | PT | 阴性 | 阴性结构 | AUC | AP | 准确率 | PT 召回 | 阴性特异度 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`dens7_rf_prior.joblib`](checkpoints/dens7_rf_prior.joblib) | 参照 | 2,620 | 672 | 1,948 | 未完整记录 | 0.973 | 0.914 | 0.921 | 0.939 | 0.914 |
| [`dens7_rf_nist.joblib`](checkpoints/dens7_rf_nist.joblib) | **主模型** | 1,353 | 672 | 681 | 181 | **0.957** | **0.961** | **0.889** | **0.898** | **0.881** |
| [`dens7_rf_gnps_mona.joblib`](checkpoints/dens7_rf_gnps_mona.joblib) | 扩展域 | 1,772 | 672 | 1,100 | 348 | 0.902 | 0.936 | 0.815 | 0.820 | 0.813 |

说明：

- 三组数据都使用同一套 PT 阳性谱（672 张，1 化合物 1 谱图），差别在阴性数据。
- 此前模型的 AUC/AP 来自原始 5 折 DENS7 RF 运行；准确率、召回和特异度来自同流程的单次 5 折 OOF 复算。NIST 与 GNPS 为 5 折×5 次重复的分组交叉验证，NIST 按 InChIKey 分组，GNPS 按 SMILES 分组。
- 表中所有指标都是**内部交叉验证**，不是新的外部测试集，也不是跨库验证。
- 阈值统一为 **0.5**；模型输出为 P(PT)。

### 1.1 为什么以 NIST 为主模型

NIST 模型是主模型，不是因为它在本表里 AUC 最高，而是因为它的训练域最接近“结构相近的高难度阴性”：

1. NIST 阴性均为 ESI 正离子、HCD、碰撞能量 20–60 eV，元数据完整；
2. 681 张阴性来自 181 个结构，阴性:阳性 = **1.01:1**，没有被 Na 加合物或离子源缺失大量稀释；
3. 在四环/五环三萜这类最难区分的阴性上，RF 仍达到 AUC **0.957**、PT 召回 **0.898**、阴性特异度 **0.881**；
4. 误报集中在四环三萜（14.2%），类别明确，交付后可以按类别复核。

因此，常规筛查建议使用 `dens7_rf_nist.joblib`。

### 1.2 三个模型为什么不能合并

三组数据的峰密度尺度不同，不能把原始 DENS7 数值直接放在同一尺度上比较，也不能把三组阴性拼成一个训练集后只交付一个模型。

关键核对：

- PT 的清洗后峰数中位在三组中均为 **292**；
- NIST 五环三萜为 **108**（仅 `[M+H]+` 为 112.5），低于 PT；
- GNPS 五环三萜为 **481**（仅 `[M+H]+` 为 452），高于 PT；
- 两边都去掉 Na 后方向仍然相反，说明差异不是由加合物单独造成，而与批次、峰提取和样本组成有关；
- GNPS 中 232 对同结构、同碰撞能量的 H/Na 配对显示：峰数 511 对 509（p=0.756），但 top-50 间距 +1.60 Da、Gini +0.131、有效峰数 −8.9。Na 加合物影响的是峰间距和强度分布，不是峰数。

因此：

- NIST 与 GNPS 分别训练、分别交付；
- 跨库只比较方向性结论，不比较绝对 DENS7 数值；
- 选择 checkpoint 时先看数据域，而不是看哪个 AUC 高。

### 1.3 此前模型与 GNPS 模型的定位

- **此前模型**：2,620 张谱图，672 PT ＋ 1,948 张普通阴性（30 张 jht ＋ 1,918 张 80MCI 高前体）。AUC 0.973 是在普通阴性上的结果，不能与 NIST 的 0.957 直接比较；它只用于说明 DENS7 信号在原始数据上的上限，不作为主交付模型。
- **GNPS＋MoNA 模型**：1,100 张阴性、348 个结构，阴性:阳性 1.64:1，覆盖五环、四环、聚醚、大环聚酮、超长脂肪链等更宽的化学空间。整体 AUC 0.902，误报 18.1%；只用 `[M+H]+` 的 696 张阴性 AUC 0.935，只用明确 ESI 的 271 张阴性 AUC 0.947。它适合做扩展域压力测试，不适合替代 NIST 主模型。

---

## 2. 快速开始

### 2.1 安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2.2 Python 调用

```python
import joblib
import numpy as np

from dens7 import dens7_from_peaks

# 一张谱：m/z 与强度一一对应；precursor_mz 为前体 m/z
mz = np.array([...], dtype=float)
intensity = np.array([...], dtype=float)
precursor_mz = 505.30

ckpt = joblib.load("checkpoints/dens7_rf_nist.joblib")
model = ckpt["model"]

features = dens7_from_peaks(mz, intensity, precursor_mz)
p_pt = float(model.predict_proba(features.reshape(1, -1))[0, 1])
print(p_pt, p_pt >= 0.5)
```

`dens7_from_peaks` 会完成去前体/同位素、top-50、归一化和 7 个特征计算；清洗后不足 50 个峰时抛出 `ValueError`。

### 2.3 批量打分 MGF

```bash
python predict.py \
  --checkpoint checkpoints/dens7_rf_nist.joblib \
  --mgf spectra.mgf \
  --out scores.tsv
```

输出列：

| 列 | 含义 |
|---|---|
| `title` | MGF 中的 TITLE |
| `precursor_mz` | 解析后的 PEPMASS |
| `n_clean` | 去前体/同位素后的峰数 |
| `p_pt` | 模型输出的 P(PT) |
| `predicted_pt` | `p_pt >= 0.5` 时为 1 |

扫描中不足 50 个清洗峰或缺少 PEPMASS 的谱会被跳过，并在命令行汇总中计数。

### 2.4 选择哪个 checkpoint

| 场景 | 建议 |
|---|---|
| NIST 类库谱、ESI 正离子、HCD、CE 20–60 eV | **`dens7_rf_nist.joblib`（默认）** |
| 原始内部数据、普通阴性背景，仅作历史对照 | `dens7_rf_prior.joblib` |
| GNPS/MoNA 类数据，阴性类别更宽、元数据不完整 | `dens7_rf_gnps_mona.joblib` |
| 不确定数据域 | 先用 NIST 主模型；同时报告 GNPS 模型的分数作为敏感性分析，不要取平均 |

### 2.5 自检

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

测试只检查特征契约、前体去除、最少峰数、checkpoint 可加载与概率范围，不替代性能评估。

---

## 3. DENS7 特征定义

特征顺序是模型契约的一部分，不能重排：

| 序号 | 特征 | 定义 |
|---:|---|---|
| 1 | `n_clean` | 去前体/同位素后的峰数；模型输入为 `log1p(n_clean)` |
| 2 | `n_above_1pct_top50` | top-50 中强度占比 >1% 的峰数 |
| 3 | `gini_top50` | top-50 强度分布的 Gini |
| 4 | `effective_peaks_entropy` | `exp(Shannon entropy)`，有效峰数 |
| 5 | `median_mz_gap` | top-50 按 m/z 排序后相邻峰间距的中位数（Da） |
| 6 | `top1_share_top50` | 最强峰占 top-50 强度的比例 |
| 7 | `top5_share_top50` | 最强 5 个峰占 top-50 强度的比例 |

预处理顺序：

1. 只保留 m/z 与强度均有限、强度 > 0 的峰；
2. m/z 完全相同的峰合并、强度相加；
3. 去掉前体及 +1/+2 同位素峰（±10 ppm）；
4. 按原始强度取 top-50；强度相同则 m/z 小者优先；
5. 除以清洗后全部峰的总强度，再取平方根；
6. 用 top-50 内部强度重新归一化后计算 2–7 项；
7. 清洗后至少 50 个峰才打分。

> `n_clean` 是 top-50 截断前的峰数，不是输入峰数，也不是 top-50 内的峰数。

---

## 4. 训练与验证

### 4.1 模型

三个 checkpoint 都是同一个随机森林配置：

```text
RandomForestClassifier(
    n_estimators=400,
    max_depth=4,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=0,
    n_jobs=1,
)
```

输入为 7 维 DENS7，正类为 PT。三个模型分别训练，不共享树、不共享标准化参数。

### 4.2 交叉验证

| checkpoint | 验证方式 | 分组键 | AUC |
|---|---|---|---:|
| prior | 5 折分层 CV，单次 | 谱图（阴性未按结构分组） | 0.973 |
| NIST | 5 折×5 次重复 `StratifiedGroupKFold` | InChIKey | 0.957 |
| GNPS＋MoNA | 5 折×5 次重复 `StratifiedGroupKFold` | SMILES | 0.902 |

NIST 的 PT 为 1 化合物 1 谱图；GNPS 的同一结构可能有不同 CE 的多张谱，因此按结构分组，避免同一结构跨折。

### 4.3 主要误报

| checkpoint | 总体误报（阈值 0.5） | 主要类别 |
|---|---:|---|
| NIST | 76/681（11.2%） | 四环三萜 45/317（14.2%）；五环三萜 28/359（7.8%） |
| GNPS＋MoNA | 199/1,100（18.1%） | 带糖五环三萜 138/533（25.9%）；四环三萜 25/191（13.1%） |

NIST 大环聚酮只有 5 张谱、1 个结构，60% 的误报率不能作为稳定估计。GNPS 的聚醚（24 张）和大环聚酮（66 张）样本量也小，类别误报率只作参考。

---

## 5. 适用边界与已知风险

### 5.1 适用

- 正离子 MS/MS，前体 m/z 已知；
- 清洗后至少 50 个峰；
- 与训练域相近的 ESI 数据；NIST 主模型更偏向 NIST 库谱条件（HCD、CE 20–60 eV）；
- 作为**第一版筛选/排序工具**，用于优先挑选需要结构复核的谱图。

### 5.2 不适用

- 不做结构鉴定，不输出具体骨架、糖基位置或化合物身份；
- 不用于负离子、低能量 CID 或清洗后不足 50 峰的谱；
- 不把 P(PT) 当作校准过的绝对概率；它没有做外部概率校准；
- 不把 NIST 模型的分数直接套到 GNPS 类数据上，也不把两库分数直接比较；
- 不承诺跨批次、跨仪器、跨样品类型泛化；目前没有前瞻性外部验证。

### 5.3 主要风险

- **类别误报**：NIST 的四环三萜、GNPS 的带糖五环三萜是主要混淆类别；交付时应保留类别和糖基标签，便于复核。
- **峰密度受采集影响**：不同批次、峰提取参数和样本组成会改变 DENS7 的绝对取值；跨库差异已在第 1.2 节量化。
- **PT 元数据缺失**：PT 的 MGF 未保留离子源、碰撞能量和仪器字段，无法逐张确认 PT 与阴性数据的采集条件完全一致。
- **小类别**：NIST 大环聚酮、GNPS 聚醚/大环聚酮/线性聚酮的样本量很小，相关误报率不稳定。
- **阈值**：0.5 是预先固定的交付阈值，未在外部测试集上重新优化；提高阈值会降低误报，同时降低 PT 召回。

---

## 6. 仓库结构

```text
dens7_pt/
├─ README.md
├─ requirements.txt
├─ requirements-dev.txt
├─ dens7.py                 # DENS7 特征计算（推理与训练共用）
├─ predict.py               # MGF 批量打分 CLI
├─ checkpoints/
│  ├─ dens7_rf_prior.joblib
│  ├─ dens7_rf_prior.json
│  ├─ dens7_rf_nist.joblib
│  ├─ dens7_rf_nist.json
│  ├─ dens7_rf_gnps_mona.joblib
│  ├─ dens7_rf_gnps_mona.json
│  └─ index.json
├─ docs/
│  └─ model_card.md
└─ tests/
   └─ test_dens7.py
```

每个 `.joblib` 是一个字典：

```python
{
    "model": RandomForestClassifier,
    "feature_names": [...],
    "decision_threshold": 0.5,
    "training": {...},
    "cross_validation": {...},
    "error_profile_at_0.5": {...},
    "intended_use": [...],
    "limitations": [...],
    ...
}
```

同名 `.json` 是模型卡，包含训练数据、交叉验证、误报、适用边界、环境版本和 SHA256。

---

## 7. 数据、代码与许可

- **训练谱图不包含在本仓库**。`data/`、`runs/`、`*.mgf` 均被 `.gitignore` 排除。
- NIST、GNPS 与 MoNA 谱图的版权和再分发条件遵循各自条款；本仓库不包含这些谱图。
- 仓库中的 `.joblib` 只包含训练后的模型参数，不包含原始谱图。
- 代码与模型参数的使用、再分发遵循课题组约定；未附 LICENSE 不代表可以自由再分发。第三方数据遵循各自条款。

---

## 8. 版本与校验

**v1.0，2026-11-02**：三个 checkpoint 冻结；NIST 为主模型；此前模型作参照；GNPS＋MoNA 用于扩展域。

| checkpoint | SHA256 |
|---|---|
| `dens7_rf_prior.joblib` | `4e63e2009745558a78db53b6bef007040ecb11bb444773f7746f5c56691dcd2d` |
| `dens7_rf_nist.joblib` | `60365c8f7d0c3ed1c63c1a6a9708877c5547bdfa8ff90830e215b0bc27950949` |
| `dens7_rf_gnps_mona.joblib` | `38323c75484a7f0731c135c8cf7d33fb788b09c6e3e779706ef1ac7501983cbd` |

科研使用请同时阅读 [`docs/model_card.md`](docs/model_card.md) 与阶段报告。模型输出只作为筛选依据，最终结构判断仍需质谱与化学复核。

课题组：华东理工大学药学院 徐志建课题组
