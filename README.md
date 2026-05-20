# Moroccan Darija Speech Synthesis with DiffWave

This repository contains the work done for the **GenAI** project on speech synthesis for **Moroccan Darija** using the **DiffWave** diffusion model and the **DODa** dataset.

The project covers a complete pipeline:

1. Audio data preprocessing
2. Training a DiffWave model **from scratch**
3. Fine-tuning a DiffWave model pre-trained on LJSpeech
4. Audio generation
5. Evaluation and comparison of the two approaches

Originally developed on **Google Colab** for GPU access, the project has been refactored into **standalone Python scripts** that run fully locally.

---

## Repository Structure

```
.
├── configs/
│   ├── default.yaml                    # all hyperparameters (single source of truth)
│   ├── train_scratch.yaml              # overrides for from-scratch training
│   └── train_finetune.yaml             # overrides for fine-tuning
│
├── src/
│   ├── config.py                       # YAML + argparse config loader
│   ├── dataset.py                      # DiffWaveDODaDataset + DataLoader factories
│   ├── diffusion.py                    # DiffusionSchedule (add_noise / p_sample / generate)
│   ├── losses.py                       # multi-resolution STFT loss
│   ├── trainer.py                      # training loop, validation, checkpointing
│   └── plotting.py                     # loss curves, waveform & mel comparisons
│
├── scripts/
│   ├── preprocess.py                   # download DODa, preprocess, save wavs + mels + CSVs
│   ├── train.py                        # train from scratch or fine-tune
│   ├── generate.py                     # generate audio from a saved checkpoint
│   └── evaluate.py                     # compare multiple runs
│
├── notebooks/                          # original Colab notebooks (kept as reference)
│   ├── 01_preprocessing_doda.ipynb
│   ├── 02_diffwave_from_scratch_training_generation.ipynb
│   ├── 03_diffwave_training_fine_tuned_10k.ipynb
│   └── 04_evaluation_resultas.ipynb
│
├── Diffwave_training_generation50k_steps.ipynb   # main 50k-step notebook
│
├── results/                            # pre-computed results (figures, audios, CSVs)
│   ├── results_10000_steps/
│   ├── results_50000_steps/
│   └── results_fine_tuning_10000/
│
├── tests/
│   └── test_pipeline.py                # CPU smoke test (no GPU, no real data needed)
│
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **DODa — Moroccan Darija Speech Dataset**, available on Hugging Face.

DODa contains audio samples in Moroccan Darija with transcriptions. Only **male voices** are used to reduce acoustic variability during training.

After filtering:

| Split | Examples |
|-------|-------:|
| Train | 6,489 |
| Val   | 721   |
| Test  | 802   |
| **Total** | **8,012** |

- ~**6.26 hours** of audio
- **3 male speakers**: M1, M2, M3
- Dataset: [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset) (public, no token required)

---

## Local Setup with `uv`

[`uv`](https://github.com/astral-sh/uv) is a fast Python package manager. It handles virtual environments and dependencies in one command.

### 1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/hadjuse/darija_project.git
cd darija_project
```

### 3. Create the virtual environment and install dependencies

```bash
uv venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

uv pip install -r requirements.txt
```

> **Note:** The DiffWave package is installed directly from GitHub — `uv` handles this automatically from `requirements.txt`.

### 4. Verify the installation

```bash
python tests/test_pipeline.py
```

This smoke test runs 10 training steps on synthetic data (CPU only, no dataset needed). Expected output:

```
==================================================
SMOKE TEST — DiffWave-Darija (CPU)
==================================================
Test 1 : DiffWaveDODaDataset.__getitem__  ✓
Test 2 : stft_loss                        ✓
Test 3 : DiffusionSchedule               ✓
Test 4 : training loop (10 steps)        ✓
Test 5 : audio generation                ✓
==================================================
Result : 5 passed, 0 failed
==================================================
```

---

## Running the Pipeline Locally

### Step 1 — Preprocess the data

Downloads DODa from Hugging Face, filters male voices, extracts mel spectrograms, and creates the train/val/test splits.

```bash
python scripts/preprocess.py --config configs/default.yaml
```

Output written to `data_preprocessed/`:

```
data_preprocessed/
├── wavs/           # 8,012 .wav files at 22,050 Hz
├── mels/           # 8,012 .npy mel spectrograms
├── train.csv
├── val.csv
├── test.csv
└── summary.json
```

> If you have a token: `python scripts/preprocess.py --hf_token hf_xxxx` or `export HF_TOKEN=hf_xxxx`.

---

### Step 2 — Train (from scratch, 50k steps)

Reproduces the best experiment from the notebook:

```bash
python scripts/train.py --config configs/train_scratch.yaml
```

Key options:

```bash
# Override any parameter inline
python scripts/train.py --config configs/train_scratch.yaml \
    --experiment_name my_run \
    --max_steps 50000

# Resume an interrupted training
python scripts/train.py --config configs/train_scratch.yaml \
    --resume runs/diffwave_doda_scratch/checkpoints/last.pt

# Fine-tune from a pre-trained LJSpeech checkpoint
python scripts/train.py --config configs/train_finetune.yaml \
    --pretrained_path /path/to/diffwave-ljspeech.pt
```

Outputs are saved in `runs/<experiment_name>/`:

```
runs/diffwave_doda_scratch/
├── checkpoints/
│   ├── best.pt      # lowest validation loss
│   ├── last.pt      # most recent periodic save
│   └── final.pt     # end of training
├── figures/
│   ├── training_loss.png
│   └── validation_loss.png
├── generated_audios/
└── results_csv/
    ├── training_losses.csv
    └── validation_losses.csv
```

---

### Step 3 — Generate audio

Generate audio samples from a saved checkpoint:

```bash
python scripts/generate.py \
    --checkpoint runs/diffwave_doda_scratch/checkpoints/best.pt \
    --num_examples 5
```

Saves `.wav` files and waveform / mel comparison figures.

---

### Step 4 — Evaluate and compare runs

```bash
python scripts/evaluate.py \
    --runs runs/diffwave_doda_scratch runs/diffwave_doda_finetune \
    --labels "From scratch 50k" "Fine-tune 10k" \
    --output_dir runs/evaluation
```

Produces overlaid training/validation curves and a summary table.

---

## Audio Configuration

| Parameter | Value |
|-----------|------:|
| Sample rate | 22,050 Hz |
| Mel bands | 80 |
| `n_fft` | 1024 |
| `hop_length` | 256 |
| `win_length` | 1024 |
| Max frequency | 8,000 Hz |
| Crop mel frames | 62 |
| Audio length per sample | 15,872 samples |

---

## Model

DiffWave with default parameters from [lmnt-com/diffwave](https://github.com/lmnt-com/diffwave):

| Parameter | Value |
|-----------|------:|
| Residual layers | 30 |
| Residual channels | 64 |
| Diffusion steps | 50 |
| Trainable parameters | 2,619,971 |

**Training setup:**
- Optimizer: Adam, lr = 2e-4
- Loss: MSE + multi-resolution STFT (weight = 0.1)
- Gradient clipping: 1.0
- AMP (float16) enabled automatically when CUDA is available

---

## Pre-computed Results

The `results/` directory contains figures, generated audios and loss CSVs from the three main experiments, so you can inspect them without rerunning training.

| Experiment | Best val loss | Best step |
|------------|:-------------:|:---------:|
| From scratch — 10k steps | 0.0326 | 10,000 |
| From scratch — 50k steps | 0.0322 | 45,000 |
| Fine-tune LJSpeech — 10k steps | 0.0412 | 5,000 |

The **50k from-scratch** run is the best model overall.

---

## Resources

- DiffWave implementation: [lmnt-com/diffwave](https://github.com/lmnt-com/diffwave)
- DiffWave paper: [Kong et al. (2021)](https://openreview.net/forum?id=a-xFK8Ymz5J)
- DODa dataset: [atlasia/DODa-audio-dataset](https://huggingface.co/datasets/atlasia/DODa-audio-dataset)
- HuggingFace Diffusion Course — Unit 4: [Audio Diffusion](https://huggingface.co/learn/diffusion-course/unit4/3)

---

## References

- Kong et al. (2021), *DiffWave: A Versatile Diffusion Model for Audio Synthesis*
- Ho et al. (2020), *Denoising Diffusion Probabilistic Models*
- Bidry et al. (2025), *DODa — Moroccan Darija Speech Dataset*
- van den Oord et al. (2016), *WaveNet: A Generative Model for Raw Audio*

```
@misc{darija_speech_dataset,
  author = {BIDRY Mahmoud, ZAIDOUNE Youssef, et al.},
  title = {Moroccan Darija Speech Dataset},
  year = {2025},
  organization = {atlasIA}
  howpublished = {Hugging Face Datasets},
  url = {https://huggingface.co/datasets/atlasia/DODa-audio-dataset-V3}
}
```
