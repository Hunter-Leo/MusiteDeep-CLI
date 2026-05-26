# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode (requires conda env with Python 3.5)
pip install -e .

# Run prediction via CLI
musitedeep -s "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" -m "py,methyl"
musitedeep --list
musitedeep -s "SEQUENCE" -m "all" -o results.json

# Run prediction directly (no CLI install)
python3 predict_multi_batch.py -input input.fasta -output output_prefix -model-prefix "models/Phosphotyrosine"

# Train CNN 10-fold ensemble
python3 train_CNN_10fold_ensemble.py -load_average_weight -balance_val -input train.fasta -output ./models/CNNmodels/ -checkpointweights ./models/CNNmodels/ -residue-types S,T -nclass=1 -maxneg 30

# Train CapsNet 10-fold ensemble
python3 train_capsnet_10fold_ensemble.py -load_average_weight -balance_val -input train.fasta -output ./models/capsmodels/ -checkpointweights ./models/capsmodels/ -residue-types S,T -nclass=1 -maxneg 30
```

## Project Overview

MusiteDeep is a deep learning-based tool for predicting post-translational modification (PTM) sites on proteins. It uses an ensemble of CNN and Capsule Network models with 10-fold cross-validation.

This fork adds a Click-based CLI (`musitedeep` command) with simplified model selection (numbers, short names, special groups like `phos`/`all`), JSON output, and interactive model listing.

## Architecture

### Prediction Pipeline (`predict_multi_batch.py`)
1. Parses FASTA input and extracts sequence fragments around target residues
2. Loads 10 CNN models + 10 CapsNet models for each selected PTM type
3. Runs batch prediction (size=500), averaging scores across all 20 models
4. Outputs tab-separated results with positions, residues, scores, and significant PTMs

### CLI Layer (`cli.py`)
- Entry point: `musitedeep` command via Click
- `resolve_models()` maps numeric/short name/group aliases to full model names
- `run_single_model_prediction()` calls `predict_multi_batch.py` per model
- `merge_results()` combines multi-model predictions, applies cutoff threshold
- Always outputs a console table; optionally writes JSON via `-o`

### Model Architectures
- **CNN** (`multiCNN_callback.py`): 3 parallel conv layers (filter sizes 1, 9, 10) with attention mechanism, batch norm, dropout
- **CapsNet** (`capsulenet_callback.py`): PrimaryCap -> CapsuleLayer (routing-by-agreement) -> Length + Mask, with custom layers in `capsulelayers.py`
- **Attention** (`attention.py`): Custom Keras layer with learned attention weights over sequence positions
- Both architectures use bootstrapping training (`Bootstrapping_multiCNN_callback.py`, `Bootstrapping_capsnet_callback.py`) with balanced negative sampling

### Data Processing (`DProcess.py`)
- `convertSampleToProbMatr()`: One-hot encodes 20 amino acids + gap character (21 categories)
- `convertRawToXY()`: Converts raw fragment data to model-ready format
- Encoding is 4D: (samples, 1, window_length, 21)

### Fragment Extraction (`EXtractfragment_sort.py`)
- `read_fasta()`: Parses FASTA, handles `#`-marked positive sites
- `extractFragforTraining()` / `extractFragforMultipredict()`: Extracts fixed-size windows around target residues

### Training (`train_CNN_10fold_ensemble.py`, `train_capsnet_10fold_ensemble.py`)
- 10-fold cross-validation split with KFold (shuffle, random_state=1234)
- Bootstrapping iteratively samples negative data (up to `maxneg` copies of positive set size)
- `calculate_avg_weights()`: Averages model weights across training steps using softmax-normalized inverse val_loss weights
- Transfer learning supported via `-inputweights` for cross-PTM weight initialization

### Models Directory Structure
```
models/<PTM type>/
  CNNmodels/     model_HDF5model_fold{0-9}_class0  (10 folds)
  capsmodels/    model_HDF5model_fold{0-9}_class0  (10 folds)
                 model_parameters  (tab-separated: nclass, window, residues)
```

## 13 Supported PTM Types

| Short Name | Full Name | Residues |
|-----------|-----------|----------|
| acetyl | N6-acetyllysine | K |
| o-glyc | O-linked_glycosylation | S,T |
| palmitoyl | S-palmitoyl_cysteine | C |
| hydroxyp | Hydroxyproline | P |
| pyrrol | Pyrrolidone_carboxylic_acid | Q |
| n-glyc | N-linked_glycosylation | N |
| ub | Ubiquitination | K |
| sumo | SUMOylation | K |
| py | Phosphotyrosine | Y |
| methyl | Methyllysine | K |
| methylr | Methylarginine | R |
| hydroxyl | Hydroxylysine | K |
| pst | Phosphoserine_Phosphothreonine | S,T |

## Important Notes

- **No test suite** exists. Verify changes by running `musitedeep` predictions and checking output correctness.
- **Model files are not in this repo**. Download from the original project and place in `models/` directory.
- **Keras 2.2.4 + TensorFlow 1.12.0** are required (not compatible with TF2.x). Use a conda environment with Python 3.5.
- During training, `CUDA_VISIBLE_DEVICES` is set in the training scripts with GPU memory fraction limits.
- The CLI sets `CUDA_VISIBLE_DEVICES=-1` to force CPU for prediction.
