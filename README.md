<div align="center">

# 🎬 Are Video Reasoning Models Ready to Go Outside?
[![arXiv](https://img.shields.io/badge/arXiv-2603.10652-b31b1b.svg)](https://arxiv.org/abs/2603.10652)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Qwen2.5-VL](https://img.shields.io/badge/Model-Qwen2.5--VL-blueviolet)](https://huggingface.co/Qwen)
[![vLLM](https://img.shields.io/badge/vLLM-0.7.2-orange)](https://github.com/vllm-project/vllm)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-ZeRO3-red)](https://www.deepspeed.ai/)
[![trl](https://img.shields.io/badge/TRL-0.16.0-yellow)](https://github.com/huggingface/trl)

---

**ROVA** is a novel training framework that improves the robustness of vision-language models for video reasoning under real-world disturbances such as weather, occlusion, and camera motion. It models a **robustness-aware consistency reward** under spatio-temporal corruptions and introduces a **difficulty-aware online training strategy** that prioritizes informative samples based on the model's evolving capability. We also introduce **PVRBench**, a new benchmark for evaluating accuracy and reasoning quality under realistic perturbations.

[**Features**](#-features) · [**Architecture**](#-architecture) · [**Quick Start**](#-quick-start) · [**Training**](#-training-pipeline) · [**Evaluation**](#-inference--evaluation) · [**Results**](#-main-results)

<img src="assets/fig2_overview.png" width="95%" alt="ROVA Framework Overview"/>

</div>

---

## 🔥 Highlights

- 🧠 **T-GRPO (Temporal Group Relative Policy Optimization)** — a novel RL algorithm that jointly optimizes over temporally-perturbed video inputs, enabling robust reasoning under frame-level corruptions
- 🌧️ **Multi-Domain Visual Corruption Engine** — realistic augmentation suite covering photometric effects (dusk, night, overexposure, shadows), weather simulation (rain, snow, hail, storm), spatial occlusion, and camera shake
- 🔄 **KL-Consistency Reward** — a dual-branch alignment signal that penalizes reasoning divergence between clean and corrupted video streams
- 🧩 **Memory-Aware Training** — intelligent sample difficulty tracking system that identifies and re-examines challenging examples across training
- ⚡ **Full-Stack Pipeline** — end-to-end support from CoT annotation → SFT warm-up → GRPO/T-GRPO reinforcement learning → multi-benchmark evaluation

---

## ✨ Features

| Capability | Description |
|:---|:---|
| **Model Support** | Qwen2.5-VL (7B) with Flash Attention 2 |
| **Training Paradigm** | SFT → GRPO / T-GRPO with DeepSpeed ZeRO-2/3 |
| **Acceleration** | vLLM-accelerated RL rollout generation |
| **Data Modality** | Image-Video mixed training |
| **Answer Types** | Multiple choice · Numerical · OCR · Free-form · Regression |
| **Corruption Types** | Photometric · Weather · Occlusion · Camera shake · Temporal drop |
| **Reward Functions** | Accuracy · Format · KL-Consistency · Length control |
| **Memory System** | Difficulty-aware sample tracking with auto-recheck |

---

## 🏗️ Architecture

```
                          ┌─────────────────────────┐
                          │     Input Video          │
                          └────────┬────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                              ▼
            ┌──────────────┐              ┌──────────────────┐
            │  Clean Path  │              │  Corrupted Path  │
            │              │              │  ┌────────────┐  │
            │              │              │  │Photometric │  │
            │              │              │  │Weather     │  │
            │              │              │  │Occlusion   │  │
            │              │              │  │Shake       │  │
            │              │              │  │Frame Drop  │  │
            │              │              │  └────────────┘  │
            └──────┬───────┘              └────────┬─────────┘
                   │                               │
                   ▼                               ▼
           ┌────────────┐                  ┌────────────┐
           │  Qwen2.5-VL│                  │  Qwen2.5-VL│
           │  (Policy)  │                  │  (Policy)  │
           └──────┬─────┘                  └──────┬─────┘
                  │                               │
                  │    ┌───────────────────┐      │
                  └───►│  T-GRPO Trainer   │◄─────┘
                       │                   │
                       │  ┌─────────────┐  │
                       │  │ R_accuracy  │  │
                       │  │ R_format    │  │
                       │  │ R_kl_cons.  │  │
                       │  │ R_length    │  │
                       │  └─────────────┘  │
                       │                   │
                       │  Memory Manager   │
                       └───────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+, CUDA-compatible GPUs (4× recommended)
- [Conda](https://docs.conda.io/) package manager

### Installation

```bash
# 1. Create environment
conda create -n rova python=3.11
conda activate rova

# 2. Install core dependencies
bash setup.sh

# 3. Install Qwen video extraction with decord acceleration
cd src/qwen-vl-utils
pip install -e .[decord]
cd ../..
```

### Install Transformers (Pinned Version)

Qwen2.5-VL undergoes frequent updates in Transformers which may cause version-related inconsistencies. Use the bundled version:

```bash
unzip transformers-main.zip
cd transformers-main
pip install .
```

### Version Requirements

| Package | Version | Note |
|:--------|:--------|:-----|
| vLLM | `0.7.2` | Required for RL acceleration |
| TRL | `0.16.0` | GRPO trainer compatibility |
| Flash Attention | latest | `pip install flash-attn --no-build-isolation` |

### Prepare Data

```bash
# Place downloaded dataset into the data directory
# then unzip:
python ./src/unzip.py
```

> 📁 Dataset should be placed in `src/r1-v/data/` before unzipping.

---

## 🎓 Training Pipeline

ROVA follows a two-stage training paradigm:

### Stage 1 — Supervised Fine-Tuning (SFT)

Warm up the model on chain-of-thought (CoT) annotated data for one epoch:

```bash
bash ./src/scripts/run_sft_video.sh
```

> 💡 To generate CoT annotations on your own data, use `src/generate_cot_vllm.py`

### Stage 2 — Reinforcement Learning (GRPO / T-GRPO)

Fine-tune the SFT checkpoint with Group Relative Policy Optimization:

```bash
# Standard GRPO with temporal corruption
bash ./src/scripts/run_grpo_video.sh

# With vLLM acceleration (recommended)
bash ./src/scripts/run_grpo_vllm_qwen25vl.sh

# With Memory-aware difficulty tracking
bash ./src/scripts/run_grpo_video_memory.sh
```

### Key Training Configurations

| Parameter | Value | Description |
|:----------|:------|:------------|
| `--temporal` | `true/false` | Toggle T-GRPO (temporal corruption) vs standard GRPO |
| `--len_control` | `true/false` | Enable/disable length control reward |
| `--num_generations` | `4-8` | Group size *G* for GRPO (↑ = lower variance, ↑ memory) |
| `--beta` | `0.04` | KL penalty coefficient |
| `--max_pixels` | `524288` | Max pixel budget per frame |
| `--max_frames` | `16` | Max video frames during training |
| `--per_device_train_batch_size` | `1` | **Must remain 1** (following R1-V convention) |

### Corruption Configuration

All corruption types are configurable via command-line arguments:

```bash
# Corruption probabilities (sum ≤ 1.0, remainder = no augmentation)
--photometric_prob 0.25    # Lighting effects: dusk / night / overexposure / shadows
--weather_prob 0.25        # Weather: rain / snow / hail / storm
--occlusion_prob 0.25      # Random block occlusion
--shake_prob 0.25          # Camera shake simulation
```

---

## 🔍 Inference & Evaluation

### Resolution Scaling

During inference, increase resolution for better performance:

| Setting | Training | Inference |
|:--------|:---------|:----------|
| Max Frame Resolution | 128 × 28 × 28 | 256 × 28 × 28 |
| Max Frames | 16 | 16 / 32 / 64 |

> Configure these in `src/qwen-vl-utils`

### Decoding Configuration

Following the official Qwen2.5-VL demo settings:

```
top_p = 0.001
temperature = 0.01
```

> ⚠️ Setting a large `top_p` may cause messy output during inference.

### Run Evaluation on All Benchmarks

```bash
# Place evaluation files in src/r1-v/Evaluation/
# Download benchmark videos and place as specified in the provided JSON files

bash ./src/eval_bench.sh
```

### Single Example Inference

```bash
python ./src/inference_example.py
```

The inference script supports all answer types through a unified prompt template:

```python
# Supported problem types
problem_type = 'multiple choice'  # → outputs single letter (A, B, C, D)
problem_type = 'numerical'        # → outputs number (42 or 3.14)
problem_type = 'OCR'              # → outputs transcribed text
problem_type = 'free-form'        # → outputs free text answer
problem_type = 'regression'       # → outputs numerical prediction
```

---

## 📊 Main Results

<div align="center">
<img src="assets/main_results.jpg" width="60%" alt="Main Results"/>
</div>

---

## 📦 Project Structure

```
robust-video-reason-main/
├── setup.sh                          # Environment setup script
├── assets/                           # Figures and visualizations
│   ├── fig2_overview.png
│   ├── dataset_demo.jpg
│   ├── main_results.jpg
│   └── fig_reward.pdf
├── src/
│   ├── inference_example.py          # Single-example inference
│   ├── eval_bench.py                 # Multi-benchmark evaluation
│   ├── eval_bench.sh                 # Evaluation launcher
│   ├── generate_cot_vllm.py          # CoT annotation generation
│   ├── unzip.py                      # Dataset extraction utility
│   ├── qwen-vl-utils/               # Qwen2.5-VL vision processing
│   ├── scripts/
│   │   ├── run_sft_video.sh          # Stage 1: SFT training
│   │   ├── run_grpo_video.sh         # Stage 2: GRPO training
│   │   ├── run_grpo_video_memory.sh  # Stage 2: GRPO + Memory
│   │   └── run_grpo_vllm_qwen25vl.sh # Stage 2: GRPO + vLLM
│   └── r1-v/
│       ├── configs/                  # DeepSpeed & training configs
│       ├── local_scripts/            # Data preparation & training
│       └── src/open_r1/
│           ├── grpo.py               # Core GRPO training logic
│           ├── grpo_baseline.py      # Baseline GRPO implementation
│           ├── grpo_memory.py        # Memory-aware GRPO
│           ├── sft_video.py          # SFT training script
│           ├── video_mask.py         # Multi-domain corruption engine
│           ├── video_mask_drop.py    # Token/pixel/frame masking
│           ├── memory_manager.py     # Difficulty-aware sample tracker
│           ├── memory_trainer.py     # Memory-integrated trainer
│           └── trainer/
│               ├── grpo_trainer.py           # Custom GRPO trainer
│               ├── grpo_trainer_baseline.py  # Baseline trainer
│               ├── grpo_trainer_v2.py        # Trainer v2
│               └── vllm_grpo_trainer_modified.py # vLLM-accelerated trainer
```

---

## 🎨 Visual Corruption Gallery

The corruption engine simulates diverse real-world visual disturbances:

| Corruption Type | Variants | Key Parameters |
|:----------------|:---------|:---------------|
| 🌅 **Photometric** | Dusk · Night · Overexposure · Shadows | `lighting_intensity` (0–1) |
| 🌧️ **Weather** | Light Rain · Heavy Rain · Storm · Snow · Hail | `particle_density`, `particle_size`, `speed` |
| 🟫 **Occlusion** | Random block masking | `mask_ratio`, `block_mean`, `block_std` |
| 📷 **Camera Shake** | Translation + Zoom + Rotation | `shake_intensity`, `zoom_range`, `smoothness` |
| ⏭️ **Temporal Drop** | Random drop · Segment drop · Keep-K | `frame_mask_ratio`, `segment_len` |

---

## 🧮 Reward Functions

ROVA employs a multi-objective reward system:

<div align="center">

📈 [**View Reward Function Visualization (PDF)**](assets/fig_reward.pdf)

</div>

| Reward | Formula | Purpose |
|:-------|:--------|:--------|
| **Accuracy** | Task-specific scoring (exact match / ROUGE / WER / relative error) | Correctness signal |
| **Format** | Regex match for `<think>...</think><answer>...</answer>` | Structural compliance |
| **KL-Consistency** | `exp(-α · KL(P_clean ‖ P_corrupt))` | Robustness alignment |
| **Length Control** | Penalizes overly long/short reasoning chains | Output quality |

---

## 🧠 Memory-Aware Training

The `MemoryManager` module tracks samples the model finds difficult:

```
┌──────────────┐    fail     ┌──────────────────┐
│  Training    │────────────►│  Memory Buffer   │
│  Sample      │             │  (max_size=100)  │
└──────────────┘             └────────┬─────────┘
                                      │
                              periodic recheck
                                      │
                              ┌───────▼─────────┐
                              │  Re-evaluate     │
                              │  ┌──pass──► Remove│
                              │  └──fail──► Keep  │
                              └──────────────────┘
```

Enable via:
```bash
--enable_sufficiency_check true \
--max_memory_size 100 \
--memory_file /path/to/memory.json
```

---

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@article{he2026rova,
  title={Are Video Reasoning Models Ready to Go Outside?},
  author={He, Yangfan and Boo, Changgyu and Yoon, Jaehong},
  journal={arXiv preprint arXiv:2603.10652},
  year={2026}
}
```

---

## 🙏 Acknowledgements

We sincerely appreciate the contributions of the open-source community, in particular the following projects:

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Base vision-language model
- [R1-V](https://github.com/Deep-Agent/R1-V) — Foundational RL training framework
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput inference engine
- [TRL](https://github.com/huggingface/trl) — Transformer reinforcement learning library
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) — Distributed training optimization

---

<div align="center">

**MIT License** · Copyright © 2026 Yangfan He, Changgyu Boo, Jaehong Yoon

Made with ❤️ for robust video understanding

</div>
