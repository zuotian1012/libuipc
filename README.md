# SoS on libuipc · 组内复现与协作

我们在 **libuipc 的 CUDA 后端**实现 Second-Order Stencil（SoS）求解，用实体和布料接触场景检查论文中的更新机制与效率提升。

这里是 `sos` 研究分支，供组内阅读代码、运行 demo 和继续开发。当前仍在核实算法一致性与收敛问题，**尚未验证论文中的加速效果**。

**先看：[中英结合的上手指南 / Getting started](README_SOS.md)**

| 你现在想做什么 | 从这里开始 |
|---|---|
| 配环境、编译、跑第一个例子 | [上手指南](README_SOS.md) |
| 看 bunny + cloth 场景 | [local_stencil_bunny_cloth.py](python/examples/local_stencil_bunny_cloth.py) |
| 看单 bunny 的 Jacobi / GS / hybrid 对比 | [local_stencil_bunny_ablation.py](python/examples/local_stencil_bunny_ablation.py) |
| 看 SoS 求解主循环 | [fem_local_stencil_solver.cu](src/backends/cuda/finite_element/fem_local_stencil_solver.cu) |
| 看局部 CCD、线搜索和接受条件 | [fem_local_stencil_linesearch.cu](src/backends/cuda/finite_element/fem_local_stencil_linesearch.cu) |
| 看论文对齐进展与历史实验 | [实现记录](agent_docs/25-sos-paper-completion.md) |
| 了解原有 libuipc 库 | [保留的原版介绍](README_LIBUIPC.md) |

## 当前做到哪里 / Status

| 模块 | 当前状态 |
|---|---|
| Local Newton-CG | 已有局部二阶方向求解及负曲率处理 |
| Hybrid sweep | 已有互补着色、GS 优先提交与 Jacobi 合并 |
| Local CCD & line search | 已有局部运动过滤与能量搜索 |
| Solid & cloth | 已接入四面体弹性、布料膜与弯曲 |
| Barriers & warm start | 已有自定义屏障、接触历史及初始化路径，规则仍在审计 |
| GPU execution | 已有 packed 方向乘积与局部 CG 实现 |
| 完整效率复现 | 尚未完成，重点排查传播慢、振荡与停止标准的可比性 |

新增 **迭代预算模式**：`--iteration-limit-policy accept_last_valid` 可在达到外层轮数上限后，以已检查的有效状态继续下一帧，并明确保留“未收敛”标记。小场景已完成 12 帧，密集场景已从检查点推进第 39–40 帧；残差仍偏大。用法见 [上手指南](README_SOS.md)，实验数据摘要见 [预算模式记录](agent_docs/27-sos-iteration-budget.md)。

已记录的 `69340034` 实现检查点通过了 202 项 stencil 断言及 120 帧布料实验。随后密集 bunny + cloth 的 build29 长跑完成 **38 帧**，在第 39 帧达到 4096 轮外层上限；这组配置的完整 120 帧落地验证尚未通过。

上述“已有实现”不等于已经与原论文逐项对齐。当前 local solver 面向无摩擦 FEM 场景，也不代表支持原库的所有刚体和摩擦功能。

## 开始协作 / Start here

```shell
git clone --branch sos --recurse-submodules https://github.com/zuotian1012/libuipc.git
cd libuipc
```

接着按 [README_SOS.md](README_SOS.md) 编译本分支。**只安装 PyPI 的 `pyuipc` 不会得到这里的修改。**

开发时从 `sos` 新建个人功能分支，PR 的目标也选 `sos`。每次实验记录 commit、完整命令、配置、停止条件和完成帧数。大型轨迹与运行库不随源码提交。

## 论文与来源 / References

- 目标论文：[Lan et al., DOI 10.1145/3592104](https://dl.acm.org/doi/10.1145/3592104)。本项目不是作者官方代码。
- 基础项目：[libuipc](https://github.com/spiriMirror/libuipc)。原库功能、引用与致谢见 [README_LIBUIPC.md](README_LIBUIPC.md)。
- 保留原有 [LICENSE](LICENSE) 和第三方归属信息。
