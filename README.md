# Federated Unsupervised Semantic Segmentation - Supplementary Material

This is the official repository of Federated Unsupervised Semantic Segmentation (https://arxiv.org/abs/2505.23292). We provide implementation details, full reproducibility instructions, and dataset preparation guidelines for the FUSS framework. All experiments in the main paper are fully reproducible using the included codebase and configuration files.

# Installation

We support installation via `pip` using a clean Python 3.12.5 environment. Two `requirements.txt` files are included:

- `requirements.txt` (default): CUDA-enabled version for systems with a compatible NVIDIA GPU.
- `requirements_cpu.txt`: Portable fallback version for CPU-only setups or unsupported platforms.

**Note:** All experiments presented in the paper were conducted on machines with CUDA 12.1-compatible NVIDIA GPUs. While a CPU-only version is available for basic usage and evaluation, full training may be impractical without GPU acceleration.

---

### Option 1: GPU Setup (Recommended)

**Requirements:**
- Python 3.12.5
- NVIDIA GPU with CUDA 12.1 drivers
- `pip ≥ 25`


**Setup:**
```bash
conda create --name fuss python=3.12.5
conda activate fuss
pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir -r requirements.txt
```

This setup matches the environment used in our experiments.

### Option 2: CPU-only Setup (Fallback)

Use this fallback only if your system does not support CUDA or lacks an NVIDIA GPU.

**Setup:**
```bash
conda create --name fuss_cpu python=3.12.5
conda activate fuss_cpu
pip install --no-cache-dir -r requirements_cpu.txt
```

Warning: Training will be extremely slow. Use this mode only for testing or evaluation on small samples without CRF post processing.

### Troubleshooting

- Dependencies such as `cupy`, `triton`, and `pydensecrf` are excluded from the CPU version for compatibility.
- If `pip install -r requirements.txt` fails, ensure your NVIDIA drivers are updated to support CUDA 12.1. Otherwise, switch to `requirements_cpu.txt`.


# Preparing the Data

## Cityscapes

The Cityscapes dataset is split using **domain-based partitioning**, as described in **Appendix C** of the paper.

For ease of use, we provide the required (empty) subfolder structure under:

`Federated_Unsupervised_Semantic_Segmentation/fuss_data`

For example, in a 3-client scenario, for client 1, you will find 6 empty city folders under:

`fuss_data/MA3_cityscapes1/gtFine/train` (one folder per domain/city)

These folders are provided in the supplementary ZIP to guide structure; users must populate them manually with images and labels from the official dataset.

**Instructions:**

1. Download the full Cityscapes dataset following the instructions in the [STEGO GitHub repository](https://github.com/mhamilton723/STEGO).
2. Replace the empty folders with the actual dataset files for the desired client split.

**Preprocessing Steps:**

Make sure you are in the `src/` directory, with the `fuss` environment activated.

```bash
# Replace {client_num} with 3, 6, or 18
python cropped_datasets_cityscapes_MA{client_num}.py
python precompute_knns_cityscapes_MA{client_num}.py
```

---

## CocoStuff27

The `fuss_data/cocostuff/` directory contains the expected layout for CocoStuff27. Please follow the same instructions as Cityscapes, to download the dataset.

**Important:**

- **Do NOT overwrite the `curated/` folder**, which contains the **1/21 subsampled dataset** used in our experiments.
- **Do NOT modify** `partitioned_splits/`, which contains the **non-i.i.d. splits** for the 18-client scenario (used in Table 2 and Appendix D).

**Preprocessing:**

From the `src/` directory, activate your environment and run:

```bash
python precompute_knns_cocostuff_MA18.py
```

---

# Training

## Cityscapes

To initiate training for the FUSS framework on Cityscapes:

```bash
python train_fed_cityscapes_MA{client_num}.py
```

The default aggregation strategy is **FedCC + Maximin**.

To modify the aggregation method:

- Open the corresponding config file:  
  `src/configs/train_config_cityscapes_MA{client_num}.yaml`
- Uncomment the desired aggregation type in the `aggregation_type:` field, to reproduce results of Table 3.

To enable additional aggregation baselines (e.g., **FedProx**, **FedMOON**, used in Table 1):

- Set the corresponding boolean flags:
  - `fedprox: true` and `fedmoon: false`
  or
  - `fedprox: false` and `fedmoon: true`

in the same `.yaml` config file.

---

## CocoStuff27

To launch **non-i.i.d.** (as described in Appendix D) federated training on CocoStuff27:

```bash
python train_fed_cocostuff_MA18.py
```

- The default setup uses **FedCC + Maximin**.
- The `train_config_cocostuff_MA18.yaml` config file includes ablation settings (Table 2).
- As with Cityscapes, set `fedprox` or `fedmoon` to `true` to enable other variants.

---

# Evaluation

To evaluate a trained model:

1. Open the corresponding evaluation config `eval_config_{dataset_name}.yaml`, and set the `model_paths` field like the CocoStuff27 example below:

```yaml
model_paths:
  - "../Federated_Unsupervised_Semantic_Segmentation/checkpoints/cocostuff/weighted_hierarchical_mm_cocostuff27/MA18_cocostuff271_exp_base_simplefedavg_date_May23_10-45-24_/agg9_state_dict.pth"
```

(Note: All intermediate folders like `checkpoints/...` will be automatically created during training.)

2. Run the evaluation script:

```bash
python eval_segmentation_{dataset_name}.py
```

Replace `{dataset_name}` with `cityscapes` or `cocostuff`.
