## Installation
To set up the project, follow these steps:
1. **Clone the repository.**
2. **Install dependencies:** We use Python 3.12+  
   Install with:
   ```bash
   pip install -r requirements.txt
   ```

## Orchestrating runs (training & inference)

There are three main entrypoints, all routed through `orchestrator.py` and driven by YAML configs under `config/`:

- **`train.py`**: trains the models listed under `models:` in `config/train.yaml` (or a custom config).
- **`inference.py`**: runs batch evaluation using `config/inference.yaml`, logging metrics, plots, and confusion matrices.

### Commands

1. **Train models listed in `config/train.yaml`:**
   ```bash
   python train.py
   ```

2. **Evaluate models defined in `config/inference.yaml`:**
   ```bash
   python inference.py
   ```

Pass `--config` to any script to point at an alternate YAML while keeping the same structure.

Each run directory contains `checkpoints/` (latest & best checkpoints), `logs/` (console outputs), and `plots/` (confusion matrix and ROC curve when labels are available). The setup targets frame-level deepfake vs. real classification but works for multiclass `ImageFolder` datasets as well.

### Per-model transform toggles

Every transform in the training and evaluation pipelines can be toggled on or off per backbone. Add a `transforms:` block under each `models.<name>` entry in the YAML config and enable the transforms you need for training/inference for each model.