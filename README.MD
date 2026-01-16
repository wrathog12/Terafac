<!--
Project: Flowers-102 — Multi-Level Image Classification
-->

# Flowers-102 — Multi-Level Image Classification

A concise and well-organized implementation of a five-level pipeline for fine-grained image classification on the Oxford Flowers-102 dataset. The project demonstrates progressive improvements using transfer learning, augmentation, attention modules, ensembling, and (optionally) deployment.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Methodology (Levels)](#methodology-levels)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Notebooks & Scripts](#notebooks--scripts)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository implements a staged approach to Fine-Grained Visual Categorization (FGVC) using the Flowers-102 dataset. Each level adds techniques that improve model performance and robustness. The goal is to provide reproducible notebooks and scripts that illustrate practical approaches for FGVC research and experimentation.

## Problem Statement

FGVC requires distinguishing many visually similar classes (102 flower species in this case) where within-class variance (lighting, pose, bloom stage) is high and between-class differences can be subtle. The project explores strategies to handle these challenges.

## Methodology (Levels)

- **Level 1 — Baseline:** Transfer learning with ResNet-50, training a new classifier head.
- **Level 2 — Augmentation & Fine-tuning:** Apply geometric and photometric augmentations and fine-tune deeper residual blocks.
- **Level 3 — Attention (CBAM):** Integrate Convolutional Block Attention Module (CBAM) to improve feature selection.
- **Level 4 — Ensemble:** Build a soft-voting ensemble combining complementary models to boost accuracy.
- **Level 5 — Deployment (optional):** Package a Gradio app for inference and simple demos, or deploy on Hugging Face Spaces.

## Repository Structure

```
├── README.MD              # This file
├── level_1/               # Baseline ResNet-50 notebook(s)
├── level_2/               # Augmentation & fine-tuning notebooks/scripts
├── level_3/               # CBAM attention implementation
├── level_4/               # Soft-voting ensemble code
├── level_5/               # Gradio/Hugging Face deployment assets (LINK)[https://huggingface.co/spaces/wrathog12/Flowers-102-ensemble]
├── models/                # Model weights are beyond 100 MB, cannot upload it through Git commands hence referencing drive Link(Link)[] (.pth)
├── results/               # Plots: accuracy curves, confusion matrices
└── requirements.txt       # Python dependencies
```

## Requirements

- Python 3.8+ recommended
- See `requirements.txt` for exact packages and versions (torch, torchvision, numpy, matplotlib, gradio, etc.)

Install with:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix / macOS
source venv/bin/activate
pip install -r requirements.txt
```

## Setup

1. Clone the repository

```bash
git clone https://github.com/yourusername/flowers102-classification.git
cd flowers102-classification
```

2. Create and activate a virtual environment (example above).
3. Install dependencies: `pip install -r requirements.txt`

### Data

The provided notebooks automatically download and prepare the Oxford Flowers-102 dataset using `torchvision.datasets` or `tensorflow_datasets`. No manual dataset download is required.

## Usage

- Open and run the notebooks in order to reproduce the experiments for each level.
- Typical workflow:
	1. Run Level 1 to train the baseline model.
	2. Continue to Level 2 to add augmentations and fine-tune.
	3. Run Level 3 to integrate attention modules.
	4. Run Level 4 to create the ensemble from saved weights.
	5. (Optional) Run Level 5 to start the Gradio demo for inference.

Start a notebook server (optional):

```bash
jupyter lab    # or `jupyter notebook`
```

## Results

Test accuracies observed during experimentation:

| Level | Methodology                         | Test Accuracy |
|-------|-------------------------------------|---------------:|
| 1     | Baseline ResNet-50                  | 88.5%         |
| 2     | Augmentation + Fine-Tuning          | 88.1%         |
| 3     | CBAM Attention Model                | 88.8%         |
| 4     | Soft-Voting Ensemble                | 90.55%        |

See `results/` for training curves, confusion matrices, and other visualizations.

## Notebooks & Scripts

- `level_1/` — notebook(s) to build the baseline model and preprocessing pipeline.
- `level_2/` — augmentation strategies and fine-tuning code.
- `level_3/` — CBAM implementation and integration into ResNet backbone.
- `level_4/` — ensemble construction and evaluation scripts.
- `level_5/` — Gradio app and deployment utilities (optional).

## Models

Trained weights are stored in the `models/` directory as `.pth` files. Use the scripts in each level to load these checkpoints for evaluation or ensembling.

## Contributing

Contributions are welcome. Suggested workflow:

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make changes and add tests or a short notebook demonstrating the change
4. Open a Pull Request describing your changes

## License

This project is provided under the MIT License. Update as needed for your use case.

## Contact

For questions or collaboration, open an issue or contact the maintainer at: your.email@example.com

---

If you'd like, I can also:

- Add badges (build, license, python version).
- Generate a short `CONTRIBUTING.md` or `requirements.txt` upgrade.
- Run a quick local check of the notebooks.
