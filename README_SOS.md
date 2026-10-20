# SoS 组内上手指南 / Collaboration guide

这份文档帮助合作同学完成三件事：**找到代码、运行场景、带着可复现的证据继续修改算法。** 中文解释流程，代码名称和常用术语保留英文，方便和论文、源码对应。

[项目首页](README.md) · [sos 分支](https://github.com/zuotian1012/libuipc/tree/sos) · [原版 libuipc 介绍](README_LIBUIPC.md)

## 1. 先了解当前边界 / Status

我们复现的是 [Lan et al., DOI 10.1145/3592104](https://dl.acm.org/doi/10.1145/3592104) 的 SoS 思路，不是作者官方代码。

目前已实现 local Newton-CG、互补着色、GS/Jacobi hybrid submission、local CCD、局部线搜索、实体与布料、自定义距离屏障、保守接触历史、inertia/barrier warm start，以及 packed GPU Hessian-vector products。

**尚未证明完整论文复现或效率提升。** 已记录的 `69340034` 实现检查点通过了 202 项 stencil 断言和 120 帧布料轨迹；随后密集 bunny/cloth build29 实验完成 38 帧，在第 39 帧达到 4096 轮外层上限。旧版本或其他配置的轨迹不能替代这组配置的验证。

现在重点检查：全局变形误差传播慢、合并更新振荡、接触历史变化导致的重启、warm start 交接，以及停止标准是否可比。当前局部求解器面向 **frictionless FEM**，不支持原库所有刚体和摩擦功能。

## 2. 克隆代码 / Clone

```shell
git clone --branch sos --recurse-submodules https://github.com/zuotian1012/libuipc.git
cd libuipc
git rev-parse HEAD
```

最后一条命令输出当前 commit，实验记录里应保存它。需要完整仓库：单独复制 stencil 文件会缺少引擎、材料、接触与配置依赖。保留原有许可证和第三方归属信息。

## 3. 编译环境 / Build from source

**必须编译本分支。`pip install pyuipc` 只安装发布版本，不包含我们的研究修改。**

本地 SoS 实验使用 Windows、CUDA 12.6、RTX 3060 Ti。下面是供合作同学使用的构建路径，并不代表已在每一种新机器上验证。

| 依赖 | 建议与说明 |
|---|---|
| GPU | NVIDIA GPU，配套兼容的驱动 |
| C++ | Windows 安装 Visual Studio C++ Build Tools，使用开发者终端 |
| CUDA Toolkit | 源码构建文档要求 >=12.4；本地实验为 12.6 |
| Python | 以下命令使用 3.11，编译和运行应保持相同小版本 |
| XMake | 本仓库要求 >=3.0.5 |

先确认 `python` 和 `python3` 都指向同一个 Python 3.11 安装，再执行：

```shell
python --version
python3 --version
xmake --version
nvcc --version
python -m pip install numpy matplotlib "polyscope>=2.5,<3" huggingface_hub
xmake f -m release --backend_cuda=true --pybind=true --python_system=true --python_version=3.11.x
xmake build -j 4 pyuipc
```

`pyuipc` 构建目标会构建所需后端，并将 Python 包复制到 `build/python/src/uipc`。其他构建方式与问题排查见 [XMake](docs/build_install/xmake.md)、[Windows/CMake](docs/build_install/windows.md)、[Linux/CMake](docs/build_install/linux.md)。

### 确认实际加载的是哪个库

在仓库根目录执行：

```shell
python -c "import sys; sys.path.insert(0, 'python/examples'); import local_stencil_beam; import uipc; print(uipc.__file__)"
python python/examples/local_stencil_bunny_cloth.py --help
```

打印的包路径应位于当前仓库的 `build/python/src/uipc` 下。示例优先使用这里的本地构建。

`UIPC_LOCAL_PACKAGE` 可指定另一个包含 `uipc/` 的构建包目录；平时不要让它残留指向旧实验运行库。修改 C++ 后，要重新构建并刷新 Python 包中的 native libraries，避免“代码改了，实际跑的还是旧 DLL”。

## 4. 先跑一个小例子 / First runs

以下命令都在仓库根目录执行，每次实验用新的输出目录。这里只检查流程能否启动，不是完整落地实验，也不是效率结论。

先跑无碰撞的小悬臂梁：

```shell
python python/examples/local_stencil_beam.py --solver local_stencil --nx 4 --frames 3 --output output/onboarding-beam
```

查看低分辨率 bunny + cloth 的网格规模，不进行仿真：

```shell
python python/examples/local_stencil_bunny_cloth.py --cloth-resolution 8 --bunny-refinement 0 --mesh-info
```

分别启动原生求解器与局部 stencil 求解器：

```shell
python python/examples/local_stencil_bunny_cloth.py --solver global --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --output output/onboarding-global
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --output output/onboarding-sos
```

**`--solver sos` 只是切换求解器，不会自动启用所有论文相关选项。** 默认仍保留 assembled products、IPC barriers、关闭 warm start 等对照配置。

要检查更多已实现机制，可以明确指定下面这组实验参数：

```shell
python python/examples/local_stencil_bunny_cloth.py --solver sos --cloth-resolution 8 --bunny-refinement 0 --dt .001 --frames 3 --hybrid-submission sequential --hessian-product packed --barrier-model sos --barrier-activation expanded --barrier-stiffness ipc --merge-acceptance local --warm-start --warm-inversion after_local --warm-max-iterations 10 --output output/onboarding-sos-custom
```

其中 expanded activation、基于 IPC 的刚度标定和有轮数上限的 warm initializer 都是当前实现选择，不能叫“原论文精确预设”。自定义屏障和原生求解器的接触目标函数也不同，不能仅用耗时直接比较算法优劣。

### 小规模定位 / Small diagnostics

现在可以先用小块测试定位问题，不必每次跑完整 bunny/cloth：

```shell
python python/examples/diagnose_local_stencil.py --output output/small-smoke
python python/examples/diagnose_local_stencil.py --resolution 2 --impact-speed .8 --output output/small-impact
```

使用当前编译的运行库（可通过 `UIPC_LOCAL_PACKAGE` 指定）。脚本逐个运行无接触、撞地、warm start 和不同合并策略，保存 `report.json` 与每组日志。原生与 SoS 的停止标准不同，耗时不能直接解释为同精度加速比。实测已发现无接触时迭代随网格加密显著增加，以及当前 warm start 在部分小场景中增加了全弹性求解工作量。独立 CPU 残差检查与下一步建议见 [小规模诊断记录](agent_docs/28-sos-small-diagnostics.md)。

### 看动画与结果 / Replay & outputs

**Budget 模式的完整场景验证失败（2026-10-18）**：密集场景每帧 32 轮的从头运行在第 57 帧出现四面体数值退化，前面已持续累积较大残差。截断接受不能保证长时间稳定；不要从已经压塌的第 56 帧继续跑，也不要删除几何检查。后端已补上近零行列式的误差余量检查，但这不是收敛修复。详见 [实验记录](agent_docs/27-sos-iteration-budget.md)。

完成一次运行后，可以回放：

```shell
python python/examples/local_stencil_bunny_cloth.py --solver sos --replay --gui --output output/onboarding-sos
```

bunny/cloth 结果位于输出目录内的 `global/` 或 `local_stencil/` 子目录。不同示例会保存以下部分或全部内容：

| 文件 | 用途 |
|---|---|
| `history.csv` | 逐帧耗时、迭代及几何指标 |
| `summary.json` | 运行总结 |
| `positions.npz` | 顶点轨迹，用于分析或回放 |
| 场景配置、checkpoint | 记录参数、恢复或同状态对照 |
| OBJ 帧 | 表面序列导出 |

达到迭代上限、缺少最终结果或只完成部分帧，不能记为“实验通过”。

## 5. 不熟悉 CUDA，从哪里读 / Code map

先读场景脚本，再沿着 `solve()` → `solve_sweep()` → `submit_sequential_hybrid()` 追踪执行顺序。分清楚 **当前位置、原始局部方向、实际提交位移、残差**：CG 求出来的方向只是候选，并不意味着已经更新了顶点。

核心文件默认位于 `src/backends/cuda/finite_element/`：

| 想理解什么 | 对应文件 |
|---|---|
| 主循环、hybrid 提交、warm 生命周期、停止条件 | `fem_local_stencil_solver.h/.cu` |
| 局部二阶问题与 Newton-CG | `local_stencil.h` |
| packed GPU 方向乘积与 CG | `local_stencil_packed.h/.cu` |
| 直接算子与 CSFD 方向导数 | `local_stencil_operator.h`、`local_stencil_csfd.h` |
| 互补着色和 GS 分组 | `fem_local_stencil_coloring.cu` |
| local CCD、线搜索、合并接受与能量 | `fem_local_stencil_linesearch.cu` |
| 实体/布料数据和导数 | `fem_local_stencil_elastic.cu`、`local_stencil_elastic*.h`、`local_stencil_snk.h` |
| 连续翻转和退化限制 | `local_stencil_inversion.h` |
| 接触装配与局部限制 | `fem_local_stencil_contact.cu`、`local_stencil_contact.h` |
| 屏障项与目标函数 | `fem_local_stencil_barriers.cu`、`local_stencil_barrier.h`、`sos_barrier.h`、`sos_contact.h` |
| 接触历史与时间步接入 | `fem_local_stencil_history.cu`、`sos_barrier_history.h/.cu` |
| 引擎调用位置 | `src/backends/cuda/engine/advance_ipc.cu` |
| 配置项注册 | `src/core/core/scene_default_config.cpp` |

算法主线可以先理解为：**局部建模 → 求候选方向 → GS 优先提交 → 剩余贡献做局部过滤并 Jacobi 合并 → 外层检查 → 残差判停**。CUDA 的线程和缓冲区细节可以放到第二遍阅读。

其他入口：

- [布料例子](python/examples/local_stencil_cloth.py)
- [单 bunny 消融](python/examples/local_stencil_bunny_ablation.py)
- [轨迹验证脚本](python/examples/validate_stencil_bunny_cloth.py)
- [配置参考](docs/specification/scene_config.md)
- [论文实现进展](agent_docs/25-sos-paper-completion.md)

`agent_docs/15–24` 记录不同历史阶段，不是同一版本的统一结论。记录中的本地 `output/` 路径不会随 Git clone 下载。

## 6. 如何分工和交结果 / Collaboration

### 迭代预算实验 / Iteration budget

可在 SoS 命令中增加 `--iteration-limit-policy accept_last_valid --max-iterations 32`。
达到 32 轮外层预算后，如果已完成 warm start 并通过正常的运动检查，将使用最后一个有效位置继续下一帧；不额外叠加一次位移。
默认 `error` 保持原来的严格行为。线搜索失败、无效几何、未完成 warm 交接等仍然报错。

截断帧保持 `converged=false`，CSV 标记 `accepted_at_iteration_limit`，并记录实际终点的两种残差；正常收敛帧这些额外终点残差列为 `NaN`，表示没有额外计算。
`summary.json` 汇总 `truncated_frames`；每帧 JSON 的 `iteration_limit` 保存终点能量等详情。
这属于近似求解实验，不是论文同精度加速结论。`--max-iterations` 限制外层轮数，每轮仍可能包含多次 `--inner-sweeps`。

### 分支与结果记录

从 `sos` 开自己的功能分支，例如：

```shell
git switch sos
git pull --ff-only origin sos
git switch -c fix/local-linesearch
```

PR 的目标选 **`sos`**，写清问题、改动规则、运行命令和验证结果。有仓库写权限的同学可以推送自己的分支；其他同学可通过 fork 提交。

建议按局部算子、提交/线搜索、接触历史/warm start、实验/文档分工。`fem_local_stencil_solver.cu` 和 `advance_ipc.cu` 是共同入口，修改前先沟通，减少冲突。

每次实验至少记录：

- commit，以及有无未提交改动；完整运行命令。
- GPU、驱动、CUDA、实际加载的运行库。
- 网格、材料、时间步、接触目标函数及相关参数。
- 停止指标与阈值、完成帧数、迭代次数和耗时。

区分 **local CG iteration、color pass、outer round**。原生 Newton 的位移判停和 SoS 的质量缩放梯度判停，不会因为阈值数字相同就具有相同精度。

`build/` 和 `output/` 已忽略。需要共享的轨迹、checkpoint 和大日志另行打包，附上配置与 commit。工作区可能含其他实验，提交时明确选择文件，不要直接用 `git add .` 全部上传。

已有局部测试位于 `apps/tests/backends/cuda/local_stencil.cu` 和 `local_stencil_cloth.cu`。根据改动验证对应行为和相关场景。C++ 仍使用 clang-format 18，CMake/XMake 的源文件归属保持同步。其余约定见 [rule.md](agent_docs/rule.md)；本分支的 SoS 首页与上手指南按组内需求使用中英结合说明。
