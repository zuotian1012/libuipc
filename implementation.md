# IPC / Contact Bottleneck Investigation — Implementation Plan

## 0. Objective

本阶段的目标不是直接提出新的 contact / collision algorithm，而是系统定位当前 IPC / libuipc pipeline 中仍然存在的 **causal bottleneck**，并判断是否存在足够 general、可复现、可解释的 research opportunity。

核心原则：

> 不只找“哪个模块最慢”，而是找“哪个机制导致大量不必要的计算发生”。

本阶段希望最终得到至少一个满足以下条件的 research seed：

- 能在至少 3 个不同场景中复现；
- 有清晰控制变量；
- 能定位到具体 pipeline 层；
- 能通过 intervention / oracle experiment 验证因果关系；
- 不是已经被 SOS / OGC / StiffGIPC / HSC / AGIPC / Barrier-Free 直接解决的问题；
- 有明显 improvement headroom。

---

# 1. Scope

优先分析以下 pipeline：

```text
geometry / motion
    ↓
broad phase
    ↓
candidate generation
    ↓
narrow phase / CCD / TOI
    ↓
contact set / active set
    ↓
contact energy / constraints
    ↓
Newton / nonlinear iteration
    ↓
linear system assembly
    ↓
PCG / preconditioner
    ↓
line search / step limiting
    ↓
state update
```

重点关注四类 failure：

1. **Correctness failure**
   - penetration
   - NaN
   - solver failure
   - iteration limit

2. **Efficiency failure**
   - runtime scaling cliff
   - Newton / PCG iteration explosion
   - candidate explosion
   - CCD query cost explosion

3. **Quality failure**
   - excessive damping
   - chatter
   - bulging
   - nonphysical repulsion

4. **Locality / uniform-treatment failure**
   - 少数 hard contacts 让整个系统承担高成本
   - 大量 collision data 被频繁重算但实际上可以复用
   - easy / hard CCD query 使用完全相同的 computation path
   - broad phase 产生大量可预测的无效 candidate

---

# 2. Experimental Infrastructure

## 2.1 Freeze baseline

- [ ] 固定 libuipc commit
- [ ] 记录 Git commit hash
- [ ] 记录 GPU 型号
- [ ] 记录 CUDA 版本
- [ ] 记录 compiler / build flags
- [ ] 固定 simulation config
- [ ] 固定 random seed（如果适用）
- [ ] 确保 benchmark 可以重复运行

输出：

```text
experiments/
  env.md
  configs/
  logs/
  plots/
  videos/
```

`env.md` 至少记录：

```text
libuipc commit:
GPU:
CUDA:
Compiler:
OS:
Build type:
Date:
```

---

## 2.2 Unified per-frame / per-iteration logging

每个 timestep 至少记录：

### Runtime

- [ ] `T_total`
- [ ] `T_broadphase`
- [ ] `T_narrowphase`
- [ ] `T_CCD`
- [ ] `T_contact`
- [ ] `T_assembly`
- [ ] `T_linear`
- [ ] `T_line_search`

### Counts

- [ ] `N_broadphase_candidates`
- [ ] `N_CCD_queries`
- [ ] `N_active_contacts`
- [ ] `N_Newton`
- [ ] `N_PCG`

### Step / convergence

- [ ] `alpha_global`
- [ ] `min_distance`
- [ ] residual norm
- [ ] Newton gradient norm
- [ ] PCG residual
- [ ] line-search iteration count

### Memory / GPU

- [ ] peak GPU memory
- [ ] optional kernel timing
- [ ] optional occupancy / bandwidth profile

推荐日志格式：

```csv
scene,frame,newton_iter,
n_candidates,n_ccd,n_active,
alpha_global,n_pcg,
t_broad,t_ccd,t_contact,t_assembly,t_linear,t_total
```

---

# 3. Phase I — Baseline Profiling

## Experiment 1 — No-contact control

### Goal

建立 solver 自身 baseline，区分：

```text
contact-induced cost
vs.
ordinary elastodynamics cost
```

### Scene

- 单个 deformable body
- 不允许发生 contact
- 与后续 contact scene 使用类似 DOF 数

### Scale

- [ ] small
- [ ] medium
- [ ] stress

建议：

```text
10k DOF
50k DOF
100k+ DOF
```

### Record

记录 Section 2.2 所有指标。

### Output

- `T_i vs DOF`
- `N_PCG vs DOF`
- `N_Newton vs DOF`

### Question

> 无 contact 时，linear solve / assembly / Newton 本身如何 scaling？

---

## Experiment 2 — Simple contact control

### Goal

建立低 contact-density baseline。

### Scene

选择一个：

- deformable block falling on plane
- cloth dropping on plane
- two deformable blocks contacting

### Sweep

保持 geometry 简单，仅改变：

```text
mesh resolution
impact velocity
```

### Output

与 no-contact control 做差：

```math
ΔT_contact = T_contact_scene - T_no_contact
```

### Question

> contact 第一次加入后，新增成本主要来自哪里？

---

# 4. Phase II — Contact Density Scaling

## Experiment 3 — Cloth stack

### Priority

**S**

### Goal

找 contact 数量增加时最先出现的 scaling cliff。

### Scene

从官方 cloth-stack sample 修改。

### Sweep A — Number of layers

```text
N_layer = 2, 4, 8, 16, 32
```

### Sweep B — Friction

```text
mu = 0, 0.2, 0.5, 1.0
```

### Sweep C — Resolution

至少两种 mesh resolution。

### Record

```text
N_candidates
N_CCD
N_active
N_Newton
N_PCG

T_broadphase
T_CCD
T_contact
T_assembly
T_linear
T_total
```

### Required plots

1. `N_active vs N_layer`
2. `N_candidates vs N_active`
3. `N_PCG vs N_active`
4. `N_Newton vs N_active`
5. 每个 runtime component vs `N_active`
6. 每个 runtime component vs `N_candidates`

### Key derived metric

```math
R_candidate =
N_candidates / N_active
```

### Interpretation

如果：

```text
N_active × 2
N_candidates × 10
```

优先调查：

```text
broad phase / candidate inflation
```

如果：

```text
N_active × 2
N_PCG × 8
```

优先调查：

```text
conditioning / solver
```

如果：

```text
N_active × 2
N_Newton × 8
```

优先调查：

```text
nonlinear progress / contact-set evolution
```

---

# 5. Phase III — Local TOI Calibration

## Experiment 4 — Local high-speed contact

### Priority

**S**

### Important

这个实验主要用于 **测量现有 baseline 的 locality / TOI behavior**，不是直接作为 novelty。

SOS 2023 已经提出 local CCD 缓解 local small TOI 导致的 global-step restriction，因此这里的目标是：

```text
quantify how severe the phenomenon is
```

以及寻找 SOS-style local CCD 仍未覆盖的问题。

### Scene

一块大 cloth / soft body，仅一个小区域发生高速 contact。

Example：

```text
------------------------------------------------
                                     ||
                                     || blade
                                     ||
------------------------------------------------
```

### Sweep A — Impact velocity

```text
v = 1, 2, 5, 10, 20
```

### Sweep B — Total system size

保持 contact region 尽量相同：

```text
10k
50k
100k
500k DOF
```

### Sweep C — Contact region size

```text
1%
2%
5%
10%
```

### Instrumentation

记录每个 CCD candidate 的：

```text
TOI / alpha_i
primitive IDs
spatial location
query cost
```

记录：

```math
alpha_global = min_i alpha_i
```

以及：

```math
argmin_i alpha_i
```

### Required statistics

```math
P(alpha_i < 0.05)
P(alpha_i < 0.1)
P(alpha_i < 0.5)
```

以及 TOI quantiles：

```text
min
p1
p10
p50
p90
p99
```

### Required plots

1. TOI histogram
2. TOI spatial map
3. limiting contact location
4. `alpha_global vs impact velocity`
5. `alpha_global vs system size`

### Interpretation

若：

```text
alpha_global << median(alpha_i)
```

说明 locality phenomenon 明显。

但不要直接得出 “local CCD 是新方向”。

下一步进入 collision-set refresh / discovery 实验。

---

# 6. Phase IV — Collision-Set Churn / Refresh

## Experiment 5 — Contact-set churn

### Priority

**S+**

### Goal

判断：

> 新 collision pairs 的产生是否具有明显 temporal / spatial coherence？

这比单纯研究 local TOI 更重要，因为 SOS Local CCD 依赖最近一次 regular CCD 已发现的 collision pairs。

### For every Newton iteration

记录 collision/contact set：

```math
C_k
```

定义：

```math
R_new =
|C_{k+1} - C_k| / |C_{k+1}|
```

以及：

```math
R_churn =
1 - |C_k ∩ C_{k+1}| / |C_k ∪ C_{k+1}|
```

### Test scenes

- [ ] slow compression
- [ ] high-speed impact
- [ ] cloth twisting
- [ ] cloth stacking
- [ ] sliding friction
- [ ] dense self-contact

### Required plots

1. `R_new vs Newton iteration`
2. `R_churn vs Newton iteration`
3. `R_churn vs velocity`
4. `R_churn vs friction`
5. `collision rediscovery time vs R_churn`

### Key questions

1. 大多数 iteration 是否只产生很少的新 collision pair？
2. 是否存在长时间 collision set 高度稳定的区域？
3. 是否只有某些局部区域频繁 churn？
4. global regular CCD 是否被低 churn 场景重复调用？

### Strong research signal

例如：

```text
90% iterations:
R_new < 2%

but global collision discovery still costs 25–40% runtime
```

可能导向：

```text
adaptive collision refresh
hierarchical collision-validity certificate
regional rediscovery
```

---

# 7. Phase V — CCD Query Difficulty Distribution

## Experiment 6 — Per-query CCD cost

### Priority

**S**

### Goal

判断是否存在：

```text
少量 hard CCD queries
消耗不成比例的 CCD runtime
```

### Instrument per CCD query

记录：

- primitive type
- relative motion magnitude
- TOI
- number of refinement / root iterations
- query runtime（如果可行）
- degeneracy indicators
- distance / angle features

### Test geometry

- normal VF
- near-parallel EE
- nearly degenerate VF
- high-speed swept contact
- thin geometry
- twisted cloth

### Required plots

1. histogram(`query cost`)
2. CDF(`query cost`)
3. top 1% / 5% queries 占 CCD runtime 比例
4. query cost vs geometric feature
5. query cost vs TOI

### Derived metrics

```math
R_1% =
runtime(top 1% queries) / total CCD runtime
```

```math
R_5% =
runtime(top 5% queries) / total CCD runtime
```

### Strong research signal

如果：

```text
1% queries consume 30%–50% CCD time
```

说明存在明显 **computational heterogeneity**。

Potential follow-up:

```text
easy / hard CCD query classification
adaptive refinement
hierarchical certificates
specialized hard-query path
```

---

# 8. Phase VI — BVH Candidate Inflation

## Experiment 7 — Broad-phase efficiency

### Priority

**S**

### Goal

判断 broad phase 是否产生大量最终无关的候选。

### Record

```text
N_BVH_pairs
N_CCD_candidates
N_active_contacts
```

定义：

```math
R_broad =
N_BVH_pairs / N_active
```

以及：

```math
R_CCD =
N_CCD_candidates / N_active
```

### Test scenes

- [ ] parallel cloth
- [ ] cloth stack
- [ ] twisted cloth
- [ ] dense packing
- [ ] long/thin geometry
- [ ] high-speed swept AABB

### Required plots

1. candidate / active ratio
2. ratio vs mesh resolution
3. ratio vs velocity
4. ratio vs contact density
5. ratio vs BVH level（若可记录）

### Strong research signal

例如：

```text
1,000,000 broad-phase candidates
10,000 active contacts

R = 100
```

并且该 inflation 在某类 geometry / motion 下系统出现。

Potential follow-up:

```text
dynamic BVH information
motion-aware pruning
contact-aware hierarchy
regional TOI bounds
```

---

# 9. Phase VII — Known-Bottleneck Exclusion Tests

这些实验主要用于排除“重新发现已有工作”。

---

## Experiment 8 — Stiffness heterogeneity

### Goal

判断慢 case 是否只是 StiffGIPC / HSC 已知问题。

### Sweep

```math
E_stiff / E_soft =
1, 10^2, 10^4, 10^6
```

### Record

```text
N_PCG
N_Newton
T_linear
T_CCD
```

### Interpretation

如果性能恶化几乎完全表现为：

```text
N_PCG explosion
```

且和 stiffness contrast 强相关，

优先认为属于：

```text
StiffGIPC / HSC territory
```

除非发现新的 root cause。

---

## Experiment 9 — Friction sweep

### Goal

寻找 friction-induced nonlinear bottleneck。

### Sweep

```text
mu = 0, 0.1, 0.3, 0.5, 1, 2
```

### Scenes

- sliding block
- cloth dragging
- cloth stack
- squeeze + sliding

### Record

```text
N_Newton
N_PCG
contact-set churn
collision-set churn
active contacts
```

### Interesting pattern

如果：

```text
mu ↑
N_Newton ↑↑
N_PCG/Newton ≈ constant
```

说明问题更偏：

```text
nonlinear / active-set evolution
```

而不是 linear conditioning。

---

## Experiment 10 — Activation distance / mesh scale

### Goal

测量 `d_hat` 与 mesh resolution 的耦合。

### Scene

近平行 cloth / cloth-plane。

定义：

```math
h = average edge length
```

扫：

```math
d_hat / h =
0.02, 0.05, 0.1, 0.2, 0.5, 1, 2
```

### Record

```text
N_candidates
N_active
N_Newton
N_PCG
equilibrium separation
```

观察：

- bulging
- chatter
- premature repulsion
- lateral force

### Purpose

主要用于：

```text
characterize parameter boundary
```

不是优先 novelty 方向。

---

# 10. Phase VIII — Under-Convergence / Iteration Budget

## Experiment 11 — Newton budget

### Goal

判断大量 global iterations 是否主要被局部 contact 拖住。

### Sweep

```text
N_Newton_max =
1, 2, 3, 5, 10, fully-converged
```

### Record

```math
E_k(t)
E_elastic(t)
E_total(t)
```

以及：

```text
trajectory difference
contact error
penetration
runtime
```

### Control

必须包含：

```text
same scene without contact
```

### Question

> 少做 global iterations 时，误差是否主要局限在 contact region？

Potential signal:

```text
large domain already converged
small contact region still changing
```

---

# 11. Phase IX — Oracle / Intervention Experiments

这是判断 **causal bottleneck** 的关键阶段。

不要追求 correctness，目标只是判断：

```text
如果某一部分“完美解决”，理论上能有多大收益？
```

---

## Experiment 12A — CCD-free oracle

### Method

在可控场景中缓存 collision result / 跳过部分 CCD。

### Compare

```text
T_original
T_no_CCD
```

### Question

> CCD 如果免费，整体 speedup 上限是多少？

---

## Experiment 12B — Linear-solve oracle

### Method

在小规模 case 中使用 direct solve 或更准确 solver。

### Record

```text
N_Newton
T_total
```

### Question

> PCG 是 root cause，还是只是被 nonlinear process 反复调用？

---

## Experiment 12C — Perfect collision/contact set

### Method

使用 converged solution 的 collision/contact set 初始化。

### Compare

```text
N_Newton_normal
N_Newton_oracle
```

### Strong signal

例如：

```text
40 → 7 Newton iterations
```

说明值得研究：

```text
contact / collision set discovery
```

---

## Experiment 12D — Perfect refresh schedule

### Method

离线记录所有真正产生新 collision pair 的 iterations。

重新运行时只在这些 iteration 做 regular collision discovery。

### Compare

```text
N_global_CCD_original
N_global_CCD_oracle
T_original
T_oracle
```

### Strong signal

例如：

```text
20 global rediscoveries
→ only 3 actually necessary
```

可能导向：

```text
adaptive collision refresh
```

这是目前特别值得关注的方向。

---

# 12. Phase X — Cross-Method Comparison

只在发现一个明确 bad regime 后做。

候选对照：

- libuipc / GIPC
- Barrier-Free
- OGC
- SOS（若有可复现实装或足够数据）

不要只比较 FPS。

根据 bottleneck 比较对应指标。

---

## If bottleneck = collision refresh

比较：

```text
# global collision detections
collision-set churn
T_collision
candidate count
```

---

## If bottleneck = hard CCD queries

比较：

```text
CCD query count
per-query cost distribution
top 1% runtime share
```

---

## If bottleneck = candidate inflation

比较：

```text
candidate / active ratio
BVH traversal cost
CCD pruning efficiency
```

---

# 13. Two-Week Minimum Plan

如果时间只有两周，严格按下面顺序做。

---

## Week 1

### Day 1–2

- [ ] 固定 baseline
- [ ] 跑通 no-contact / simple-contact / cloth-stack
- [ ] 完成统一 profiler
- [ ] 输出所有基本 runtime + iteration 数据

### Day 3

- [ ] Dense contact scaling
- [ ] 画 candidate / active / PCG / Newton scaling

### Day 4

- [ ] Local high-speed contact
- [ ] 导出所有 TOI
- [ ] 画 TOI distribution

### Day 5–6

- [ ] 实现 collision-set logging
- [ ] 计算 `R_new`
- [ ] 计算 `R_churn`

### Day 7

- [ ] 汇总第一周结果
- [ ] 选择最异常的两个 regime

---

## Week 2

### Day 8–9

- [ ] CCD per-query difficulty profiling
- [ ] 计算 top 1% / top 5% runtime contribution

### Day 10

- [ ] BVH candidate inflation
- [ ] 分析 candidate / active ratio

### Day 11–12

- [ ] Oracle collision-set experiment
- [ ] Oracle refresh schedule experiment

### Day 13

- [ ] 做 stiffness / friction exclusion test

### Day 14

- [ ] 汇总
- [ ] 选择一个 research seed
- [ ] 写 3-slide internal report

---

# 14. Required Final Deliverable

两周后不要提交几十张图。

只提交最多 5 页 / 3 slides。

---

## Slide 1 — Failure

必须是一句明确结论：

Example：

> 92% of Newton iterations introduce fewer than 2% new collision pairs, yet global collision discovery is still repeated every iteration and consumes 31% of runtime.

或者：

> 1.4% of CCD queries account for 43% of total CCD time in high-speed thin-shell contact.

或者：

> Broad-phase candidate count grows 17× while active contacts grow only 2.3× in dense near-parallel cloth contact.

---

## Slide 2 — Cause

用 intervention 证明原因。

Example：

```text
baseline                       420 ms
oracle collision set          190 ms
oracle refresh schedule       155 ms
```

说明：

```text
bottleneck ≠ CCD kernel speed
bottleneck = unnecessary rediscovery
```

---

## Slide 3 — Research Question

把现象压缩成数学 / algorithmic question。

Examples：

```text
Can collision information be certified locally so that
global collision rediscovery is performed only where
and when new contacts can actually emerge?
```

或：

```text
Can heterogeneous CCD queries be classified and solved
with different refinement strategies while preserving
conservative collision guarantees?
```

或：

```text
Can a BVH provide conservative regional validity intervals
for collision sets, avoiding repeated full-resolution
collision discovery?
```

---

# 15. Kill Criteria

一个方向出现以下情况时优先停止：

- [ ] 只在一个特殊 geometry 中出现
- [ ] 不能随控制变量稳定复现
- [ ] speedup theoretical upper bound < 1.2×
- [ ] 已被 SOS / OGC / StiffGIPC / HSC / AGIPC / Barrier-Free 直接解决
- [ ] 只能靠参数 tuning 改善
- [ ] 没有可解释 root cause
- [ ] 只能得到 kernel-level 小优化
- [ ] 不能构造 clean toy example

---

# 16. Continue Criteria

一个方向值得继续至少 1–3 个月，如果：

- [ ] 至少 3 个场景复现
- [ ] 有一个清晰控制变量
- [ ] 有明显 scaling cliff
- [ ] 可以通过 oracle / intervention 让问题基本消失
- [ ] bottleneck 具有明显 generality
- [ ] baseline 已经较强
- [ ] 现有方法没有直接覆盖
- [ ] 可以用一句话解释 failure mechanism
- [ ] 可以画出一张 reviewer 一眼看懂的关键图
- [ ] 存在明显的 algorithmic headroom

---

# 17. Research Principle

整个阶段始终遵守：

```text
不要：
profile → 找最慢模块 → 优化 kernel
```

而要：

```text
parameter sweep
    ↓
find scaling cliff
    ↓
measure work quantity + unit cost
    ↓
intervention / oracle
    ↓
identify causal bottleneck
    ↓
find system heterogeneity / unnecessary work
    ↓
formulate research question
    ↓
only then design algorithm
```

最终想找到的不是：

> “CCD 比较慢。”

而是：

> “为什么这些 CCD / collision computations 本来不应该发生，却仍然在发生？”

这才是更可能形成 paper 的问题。
