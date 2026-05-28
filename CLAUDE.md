# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode (requires conda env with Python 3.5)
pip install -e .

# View available models with biological descriptions
musitedeep --list

# View all options
musitedeep --help

# Predict with specific models (numbers, short names, or groups)
musitedeep -s "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" -m "py,methyl"
musitedeep -s "SEQUENCE" -m "9,10" --cutoff 0.5

# Phosphorylation prediction (py + pst) or all models
musitedeep -s "SEQUENCE" -m "phos"
musitedeep -s "SEQUENCE" -m "all"

# Save results to JSON
musitedeep -s "SEQUENCE" -m "all" -o results.json

# Custom cutoff threshold (default 0.5)
musitedeep -s "SEQUENCE" -m "py" -c 0.3

# Run prediction directly (no CLI install)
python3 predict_multi_batch.py -input input.fasta -output output_prefix -model-prefix "models/Phosphotyrosine"

# Train CNN 10-fold ensemble
python3 train_CNN_10fold_ensemble.py -load_average_weight -balance_val -input train.fasta -output ./models/CNNmodels/ -checkpointweights ./models/CNNmodels/ -residue-types S,T -nclass=1 -maxneg 30

# Train CapsNet 10-fold ensemble (transfer learning from pst to py)
python3 train_capsnet_10fold_ensemble.py -load_average_weight -balance_val -inputweights ./models/Phosphoserine_Phosphothreonine/capsmodels/model_HDF5model_fold0_class0 -input train.fasta -output ./models/capsmodels/ -checkpointweights ./models/capsmodels/ -residue-types Y -nclass=1 -maxneg 30
```

## Project Overview

MusiteDeep is a deep learning-based tool for predicting post-translational modification (PTM) sites on proteins. It uses an ensemble of CNN and Capsule Network models with 10-fold cross-validation (20 models total per PTM type).

This fork adds a Click-based CLI (`musitedeep` command) with simplified model selection (numbers, short names, special groups like `phos`/`all`), JSON output, and interactive model listing.

## Architecture

### CLI Layer (`cli.py`)
- Entry point: `musitedeep` command via Click
- `check_model_data()`: Validates that model directories exist before running
- `resolve_models()`: Maps numeric/short name/group aliases to full model names (supports `phos` → py + pst, `all` → all 13 types)
- `run_single_model_prediction()`: Writes sequence to temp FASTA, spawns `predict_multi_batch.py` as subprocess per model, parses results
- `merge_results()`: Combines multi-model predictions, applies cutoff threshold (default 0.5), sorts by position
- `parse_results()`: Parses tab-separated MusiteDeep output into structured dicts
- Always outputs a console table; optionally writes JSON via `-o`
- Sets `CUDA_VISIBLE_DEVICES=-1` (CPU only)

### Prediction Pipeline (`predict_multi_batch.py`)
1. Parses FASTA input and extracts sequence fragments around target residues via `extractFragforMultipredict()`
2. Loads model parameters (nclass, window size, residues) from `models/<PTM>/capsmodels/model_parameters`
3. Loads 10 CNN models + 10 CapsNet models per PTM type
4. Runs batch prediction (batch_size=500), averaging scores across all 20 models
5. Outputs tab-separated results with positions, residues, all PTM scores, and significant PTMs (cutoff=0.5)

### CNN Architecture (`multiCNN_callback.py`)
- 3 parallel convolutional paths: filter sizes 1, 9, 10 (filters: 200, 150, 200)
- Attention mechanism (`attention.py`) over sequence positions with learned weights
- Batch normalization, dropout (0.75), ReLU activations
- Dense layers: 149 → 8 → 2 (softmax output)
- Adam optimizer, he_normal initialization

### CapsNet Architecture (`capsulenet_callback.py`)
- PrimaryCap: Conv2D → Reshape → Squash (generates initial capsules)
- CapsuleLayer: Dynamic routing-by-agreement between capsule layers
- Length layer: Computes vector lengths for classification probability
- Mask layer: Reconstruction regularization via decoder
- Custom layers in `capsulelayers.py`: `PrimaryCap`, `CapsuleLayer`, `CapsuleLayer_nogradient_stop`, `Length`, `Mask`
- Adam optimizer

### Attention Mechanism (`attention.py`)
- Custom Keras layer with learned attention weights over sequence positions
- Fully connected layer learns per-position importance scores
- Softmax-normalized attention weights applied to CNN feature maps

### Bootstrapping Training (`Bootstrapping_multiCNN_callback.py`, `Bootstrapping_capsnet_callback.py`)
- Separates positive and negative samples
- Iteratively samples negative data: each iteration draws `slength` negative samples (up to `maxneg` copies of positive set size)
- Stage 1: Train on balanced subset for `nb_epoch1` epochs
- Stage 2: Iteratively add negative samples, retraining for `nb_epoch2` epochs each round
- `calculate_avg_weights()`: Averages model weights across training steps using softmax-normalized inverse val_loss weights
- Transfer learning via `-inputweights` for cross-PTM weight initialization

### Data Processing (`DProcess.py`)
- `convertSampleToProbMatr()`: One-hot encodes 20 amino acids + gap character (21 categories), padding unknown residues ('X') with zeros
- `convertRawToXY()`: Converts raw fragment DataFrame to 4D numpy array
- Encoding shape: (samples, 1, window_length, 21) — channels_last format

### Fragment Extraction (`EXtractfragment_sort.py`)
- `read_fasta()`: Parses FASTA, handles `#`-marked positive sites
- `extractFragforTraining()` / `extractFragforMultipredict()`: Extracts fixed-size windows around target residues
- Handles sequence boundaries by padding with '-' characters

### Model Checkpointing (`LossCheckPoint.py`)
- Custom `LossModelCheckpoint` Keras callback that saves model weights at epoch end
- Tracks validation loss to determine best weights

### Entry Points (`setup.py`)
- Console script: `musitedeep=cli:predict`
- Dependencies: click, numpy, scipy, scikit-learn, pillow, h5py, pandas, keras==2.2.4, tensorflow==1.12.0
- Python >= 3.5 required

### CLI Prediction Flow (end-to-end)
1. `musitedeep -s SEQ -m py,methyl -o out.json`
2. `cli.py`: Validates sequence, resolves "py" → "Phosphotyrosine", "methyl" → "Methyllysine"
3. Writes sequence to temp FASTA file
4. For each model, spawns: `python3 predict_multi_batch.py -input temp.fasta -output temp_prefix -model-prefix "models/Phosphotyrosine"`
5. Each `predict_multi_batch.py` run loads 20 models (10 CNN + 10 CapsNet), runs batch prediction
6. CLI parses and merges all results, applies cutoff, outputs table + optional JSON
7. Cleans up temp files

## 13 Supported PTM Types

| # | Short Name | Full Name | Residues | Biological Description |
|---|-----------|-----------|----------|----------------------|
| 1 | acetyl | N6-acetyllysine | K | Gene expression regulation, chromatin remodeling |
| 2 | o-glyc | O-linked_glycosylation | S,T | Protein folding, cell adhesion, immune response |
| 3 | palmitoyl | S-palmitoyl_cysteine | C | Membrane association, protein trafficking |
| 4 | hydroxyp | Hydroxyproline | P | Collagen stability, extracellular matrix formation |
| 5 | pyrrol | Pyrrolidone_carboxylic_acid | Q | Protein degradation signal, N-terminal processing |
| 6 | n-glyc | N-linked_glycosylation | N | Protein folding, quality control, cell recognition |
| 7 | ub | Ubiquitination | K | Protein degradation, cell cycle, DNA repair |
| 8 | sumo | SUMOylation | K | Nuclear transport, transcriptional regulation |
| 9 | py | Phosphotyrosine | Y | Signal transduction, cell growth, differentiation |
| 10 | methyl | Methyllysine | K | Gene expression, chromatin structure regulation |
| 11 | methylr | Methylarginine | R | Gene expression, RNA processing, DNA repair |
| 12 | hydroxyl | Hydroxylysine | K | Collagen cross-linking, connective tissue stability |
| 13 | pst | Phosphoserine_Phosphothreonine | S,T | Signal transduction, metabolic regulation |

Special groups: `phos` = py + pst (phosphorylation), `all` = all 13 types.

## Models Directory Structure

```
models/<PTM type>/
  CNNmodels/     model_HDF5model_fold{0-9}_class0  (10 folds)
  capsmodels/    model_HDF5model_fold{0-9}_class0  (10 folds)
                 model_parameters  (tab-separated: nclass, window, residues)
```

## Project Files

| File | Purpose |
|------|---------|
| `cli.py` | Click CLI entry point (`musitedeep` command) |
| `predict_multi_batch.py` | Core prediction pipeline (loads models, runs batch inference) |
| `multiCNN_callback.py` | CNN model architecture definition |
| `capsulenet_callback.py` | Capsule Network model architecture definition |
| `capsulelayers.py` | Custom Keras layers: PrimaryCap, CapsuleLayer, Length, Mask |
| `attention.py` | Custom attention layer for CNN |
| `LossCheckPoint.py` | Custom model checkpoint callback |
| `Bootstrapping_multiCNN_callback.py` | CNN bootstrapping training logic |
| `Bootstrapping_capsnet_callback.py` | CapsNet bootstrapping training logic |
| `DProcess.py` | Data encoding (one-hot, fragment→tensor conversion) |
| `EXtractfragment_sort.py` | FASTA parsing and sequence fragment extraction |
| `train_CNN_10fold_ensemble.py` | CNN 10-fold cross-validation training |
| `train_capsnet_10fold_ensemble.py` | CapsNet 10-fold cross-validation training |
| `setup.py` | Package setup with console_scripts entry point |

## Important Notes

- **No test suite** exists. Verify changes by running `musitedeep` predictions and checking output correctness.
- **Model files are not in this repo**. Download from the original project and place in `models/` directory.
- **Keras 2.2.4 + TensorFlow 1.12.0** are required (not compatible with TF2.x). Use a conda environment with Python 3.5.
- During training, `CUDA_VISIBLE_DEVICES` is set in training scripts with `per_process_gpu_memory_fraction=0.3` for CNN and `0.5` for CapsNet.
- The CLI sets `CUDA_VISIBLE_DEVICES=-1` to force CPU-only mode for prediction.
- The `testdata/` directory contains training/validation FASTA files for all PTM types (see original repo).
- Default prediction cutoff is 0.5; adjust with `-c` for more sensitive or specific detection.
