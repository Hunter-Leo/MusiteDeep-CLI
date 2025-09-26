# MusiteDeep-CLI Installation Guide

## Prerequisites

### 1. Install Conda Package Manager
You need a conda package manager installed. Choose one of the following (Miniforge is recommended):

- **Miniforge (Recommended)**: https://github.com/conda-forge/miniforge#install
- **Miniconda**: https://docs.conda.io/en/latest/miniconda.html
- **Anaconda**: https://docs.anaconda.com/anaconda/install/

### 2. Clone This Project
```bash
# Clone the repository
git clone https://github.com/Hunter-Leo/MusiteDeep-CLI.git

# Enter project directory
cd MusiteDeep-CLI
```

### 3. Download Model Data
⚠️ **Important**: This repository does not include model files. Download them first:

```bash
# Download models from the original repository
# Visit: https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep/models
# Place all model folders in the models/ directory of this project
```

## Environment Setup

### 4. Create conda environment
```bash
mamba create -n musitedeep python=3.5.2 -y
```

### 5. Install dependencies
```bash
# Activate environment
conda activate musitedeep

# Install grpcio first
mamba install -c conda-forge grpcio

# Install Python packages using Tsinghua mirror
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy scipy scikit-learn pillow h5py pandas keras==2.2.4 tensorflow==1.12.0
```

## CLI Installation

### 6. Install MusiteDeep CLI
```bash
# Navigate to the project directory
cd /path/to/MusiteDeep

# Install CLI tool in development mode
pip install -e .
```

### 7. Verify Installation
```bash
# Test the CLI tool
musitedeep --list

# Or use with conda run from any environment
conda run -n musitedeep musitedeep --list
```

## Usage Examples

### Basic Usage
```bash
# View available models
musitedeep --list

# Predict with specific models (using numbers)
musitedeep -s "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" -m "9,10"

# Predict with short names
musitedeep -s "PROTEIN_SEQUENCE" -m "py,methyl"

# Phosphorylation prediction
musitedeep -s "PROTEIN_SEQUENCE" -m "phos"

# All models with JSON output
musitedeep -s "PROTEIN_SEQUENCE" -m "all" -o "results.json"
```

### Advanced Options
```bash
# Custom cutoff threshold
musitedeep -s "SEQUENCE" -m "py" -c 0.3

# Output to nested directory (auto-created)
musitedeep -s "SEQUENCE" -m "all" -o "output/results/prediction.json"
```

## Troubleshooting

### Model Files Missing
If you get model-related errors, ensure you have downloaded all model folders from the original repository and placed them in the `models/` directory.

### Path Issues
The CLI tool can be used from any directory. If you encounter path issues, try using the full conda run command:
```bash
conda run -n musitedeep musitedeep [options]
```
