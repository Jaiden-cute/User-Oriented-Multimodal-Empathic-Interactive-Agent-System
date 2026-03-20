# User-Oriented Multimodal Empathic Interactive Agent System
面向用户的多模态共情交互智能体系统

本项目旨在构建一个面向用户的多模态共情交互智能体系统，结合语音、文本等多种输入形式，对用户的意图、情绪和交互状态进行综合感知，并通过统一决策模块生成更具个性化和共情能力的交互反馈。

系统主要包括用户输入处理、多模态感知、意图识别、多模态决策和结果输出五个核心部分，最终服务于 App 端的人机交互场景。

## System Architecture

系统整体流程如下：

用户 → 输入模块 → 感知模块 / 意图识别模块 → 多模态决策模块 → 输出模块 → App

## Project Structure

```text
.
├── IAMM_Model/          # 多模态决策模块
├── Intent_analysis/     # 意图识别模块
├── Perception_Model/    # 多模态感知模块
├── UI/                  # 前端 / App 交互界面
├── docs/                # 项目文档与架构图（可选）
└── README.md
