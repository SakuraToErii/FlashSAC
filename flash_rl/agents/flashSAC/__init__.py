"""FlashSAC 算法实现包。

对应论文: Kim et al., "FlashSAC: Fast and Stable Off-Policy Reinforcement
Learning for High-Dimensional Robot Control" (arXiv:2604.04539).

模块划分与论文结构对应:
  - network / layer  → §4.2 架构 (Figure 2) 与 §3.2 SAC 策略头
  - update           → §3.2 目标 (2)(4)(5) 与 §4.2 稳定化技巧
  - agent            → §4.1 数据吞吐 / §4.3 探索 的运行时编排
"""
