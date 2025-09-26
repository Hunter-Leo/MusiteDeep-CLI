# MusiteDeep-CLI
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.5+](https://img.shields.io/badge/python-3.5+-blue.svg)](https://www.python.org/downloads/)
[![GitHub issues](https://img.shields.io/github/issues/Hunter-Leo/MusiteDeep-CLI)](https://github.com/Hunter-Leo/MusiteDeep-CLI/issues)
[![GitHub stars](https://img.shields.io/github/stars/Hunter-Leo/MusiteDeep-CLI)](https://github.com/Hunter-Leo/MusiteDeep-CLI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Hunter-Leo/MusiteDeep-CLI)](https://github.com/Hunter-Leo/MusiteDeep-CLI/network)

## Project Modifications

This project is based on the original [MusiteDeep_web](https://github.com/duolinwang/MusiteDeep_web) and adds the following enhancements:

### Key Features Added
- **Command Line Interface (CLI)**: A user-friendly `musitedeep` command for easy PTM prediction
- **Simplified Model Selection**: Support for numbers, short names, and special options (e.g., `--models 1,2,3` or `--models py,methyl`)
- **JSON Output Format**: Structured prediction results with risk assessment
- **Interactive Model List**: `--list` option to display available PTM models with biological descriptions
- **Installation Guide**: Added [INSTALL.md](INSTALL.md) for easy setup instructions

## CLI Usage Examples

### View Available Models
```bash
musitedeep --list
```
```
Available PTM Models:
========================================================================================================================
No.  Short Name      Full Name                      Biological Description                            
------------------------------------------------------------------------------------------------------------------------
1    acetyl          N6-acetyllysine                Gene expression regulation, chromatin remodeling  
2    o-glyc          O-linked_glycosylation         Protein folding, cell adhesion, immune response   
3    palmitoyl       S-palmitoyl_cysteine           Membrane association, protein trafficking         
4    hydroxyp        Hydroxyproline                 Collagen stability, extracellular matrix formation
5    pyrrol          Pyrrolidone_carboxylic_acid    Protein degradation signal, N-terminal processing 
6    n-glyc          N-linked_glycosylation         Protein folding, quality control, cell recognition
7    ub              Ubiquitination                 Protein degradation, cell cycle, DNA repair       
8    sumo            SUMOylation                    Nuclear transport, transcriptional regulation     
9    py              Phosphotyrosine                Signal transduction, cell growth, differentiation 
10   methyl          Methyllysine                   Gene expression, chromatin structure regulation   
11   methylr         Methylarginine                 Gene expression, RNA processing, DNA repair       
12   hydroxyl        Hydroxylysine                  Collagen cross-linking, connective tissue stability
13   pst             Phosphoserine_Phosphothreonine Signal transduction, metabolic regulation         
------------------------------------------------------------------------------------------------------------------------

Special options:
  phos    - Phosphorylation (py + pst)
  all     - All models

Usage examples:
  --models 1,2,3      (use numbers)
  --models py,methyl  (use short names)
  --models phos       (phosphorylation)
```

### Basic Prediction
```bash
musitedeep -s "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" -m "py,methyl"
```
```
Running prediction for model: Phosphotyrosine
Running prediction for model: Methyllysine

Prediction Results:
------------------------------------------------------------
Position Residue  PTM Type                  Score    Risk  
------------------------------------------------------------
2        K        Methyllysine              0.825    HIGH  
10       K        Methyllysine              0.415    LOW   
20       K        Methyllysine              0.080    LOW   
45       Y        Phosphotyrosine           0.180    LOW   
51       Y        Phosphotyrosine           0.108    LOW   
60       Y        Phosphotyrosine           0.271    LOW   
------------------------------------------------------------
Total predictions: 6
High risk predictions: 1
```

### Using Model Numbers
```bash
musitedeep -s "PROTEIN_SEQUENCE" -m "9,10"  # Phosphotyrosine + Methyllysine
```

### Phosphorylation Analysis
```bash
musitedeep -s "PROTEIN_SEQUENCE" -m "phos"  # All phosphorylation types
```

### Save Results to JSON
```bash
musitedeep -s "PROTEIN_SEQUENCE" -m "all" -o "results/prediction.json"
```
```json
{
  "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
  "models_used": ["Phosphotyrosine", "Methyllysine"],
  "cutoff": 0.5,
  "predictions": [
    {
      "position": 2,
      "residue": "K",
      "ptm_type": "Methyllysine",
      "score": 0.825,
      "risk": true
    },
    {
      "position": 10,
      "residue": "K", 
      "ptm_type": "Methyllysine",
      "score": 0.415,
      "risk": false
    }
  ]
}
```

### Custom Cutoff Threshold
```bash
musitedeep -s "PROTEIN_SEQUENCE" -m "py" -c 0.3  # Lower threshold for more sensitive detection
```

### Important Notes
⚠️ **Model Data Not Included**: This repository does not contain the pre-trained model files. You need to download them from the original project:

**Download Models**: https://github.com/duolinwang/MusiteDeep_web/tree/master/MusiteDeep/models

**Installation**: Place the downloaded model folders in the `models/` directory of this project.

### Repository Links
- **Original Project**: https://github.com/duolinwang/MusiteDeep_web
- **This Project**: https://github.com/Hunter-Leo/MusiteDeep-CLI.git

---

# Original MusiteDeep Documentation

# PTM prediction

We provide 13 different models for PTM site predictions and customized model training that enables users to train models by their own data. 
Users can find the models in the folder of MusiteDeep/models. 
##### Installation

  - Installation has been tested in Ubuntu 16.04.5 LST and Mac OS HighSierra with python 3.5.2: 
You can install the dependent packages by the following commands:
    ```sh
    sudo apt-get install -y python3.5 python3-pip
    python3 -m pip install numpy 
    python3 -m pip install scipy
    python3 -m pip install scikit-learn
    python3 -m pip install pillow
    python3 -m pip install h5py
    python3 -m pip install pandas
    python3 -m pip install keras==2.2.4
    python3 -m pip install tensorflow==1.12.0 (or install the GPU supported tensorflow by pip3 install tensorflow-gpu==1.12.0 refer to https://www.tensorflow.org/install/ for instructions)
    ```
    Download the stand-alone tool by:
    ```sh
    git clone https://github.com/duolinwang/MusiteDeep_web
    ```
##### Running on GPU or CPU
>If you want to use GPU, you also need to install [CUDA]( https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn); refer to their websites for instructions. 
CPU is only suitable for prediction not training. 

##### For general users who want to perform PTM site prediction by our provided model :
go to the MusiteDeep folder which contains the predict_multi_batch.py
```sh
python3 predict_multi_batch.py -input [custom prediction data in FASTA format] -output [custom specified prefix for the prediction results] -model-prefix [prefix of pre-trained model] 
```
For details of the parameters, use the -h or --help parameter.

The -model-prefix can be "models/XX/", here XX represents one pre-trained model in the folder of "MusiteDeep/models/". To predict for multiple PTMs, use ";" to separate the prefixes of different pre-trained models.
For example, to predict for phosphotyrosine and methyllysine simultaneously, run the following command:
```sh
python3 predict_multi_batch.py -input testdata/Phosphorylation/Y/test_allspecies_sequences.fasta -output test/output  -model-prefix "models/Phosphotyrosine;models/Methyllysine";

or

python3 predict_multi_batch.py -input testdata/Phosphorylation/Y/test_allspecies_sequences.fasta -output test/output  -model-prefix "models/Phosphotyrosine;Methyllysine";
```

##### For advanced users like to perform training and prediction by using their own data
Because we used ensemble models by two deep-learning architectures, two types of models need to be trained: one is the CNN model [1] trained by train_CNN_10fold_ensemble.py, and the other is the capsule model [2] trained by train_capsnet_10fold_ensemble.py. To train a customized predictor, users can run the following commands and replace with their own data and parameters.
```sh
python3 train_CNN_10fold_ensemble.py -load_average_weight -balance_val -input [custom training data in FASTA format] -output [folder for the output models] -checkpointweights [folder for the intermediate checkpoint files] -residue-types [custom specified residue types]

python3 train_capsnet_10fold_ensemble.py -load_average_weight -balance_val -input [custom training data in FASTA format] -output [folder for the output model] -checkpointweights [folder for the intermediate checkpoint files] -residue-types [custom specified residue types]
```
The training data should be in the FASTA format. Residues followed by "#" indicates the positive sites, residues in the custom specified residue types but without "#" are considered as the negative sites. Tje -residue-types parameter indicates the potential modification residue types that this model focuses on. Multiple types of residues are separated with ','. And all the residues specified by this parameter will be trained in one predictor. For details of other parameters, use the -h or --help parameter.
##### Examples of commands used to train our provided models:
 For Phosphoserine_Phosphothreonine:

 ```sh
 python3 train_CNN_10fold_ensemble.py -load_average_weight -balance_val -input "testdata/Phosphorylation/ST/train_allspecies_sequences_annotated.fasta" -output "./models_test/Phosphoserine_Phosphothreonine/CNNmodels/" -checkpointweights "./models_test/Phosphoserine_Phosphothreonine/CNNmodels/" -residue-types S,T -nclass=1 -maxneg 30
 python3 train_capsnet_10fold_ensemble.py -load_average_weight -balance_val -input "testdata/Phosphorylation/ST/train_allspecies_sequences_annotated.fasta" -output "./models_test/Phosphoserine_Phosphothreonine/capsmodels/" -checkpointweights "./models_test/Phosphoserine_Phosphothreonine/capsmodels/" -residue-types S,T -nclass=1 -maxneg 30
```
 For Phosphotyrosine, we transferred the pre-trained weights from Phosphoserine_Phosphothreonine:
```sh
 python3 train_CNN_10fold_ensemble.py -load_average_weight -balance_val -inputweights ./models/Phosphoserine_Phosphothreonine/CNNmodels/model_HDF5model_fold0_class0 -input "testdata/Phosphorylation/Y/train_allspecies_sequences_annotated.fasta" -output "./models_test/Phosphotyrosine/CNNmodels/" -checkpointweights "./models_test/Phosphotyrosine/CNNmodels/" -residue-types Y -nclass=1 -maxneg 30
 python3 train_capsnet_10fold_ensemble.py -load_average_weight -balance_val -inputweights ./models/Phosphoserine_Phosphothreonine/capsmodels/model_HDF5model_fold0_class0 -input "testdata/Phosphorylation/Y/train_allspecies_sequences_annotated.fasta" -output "./models_test/Phosphotyrosine/capsmodels/" -checkpointweights "./models_test/Phosphotyrosine/capsmodels/" -residue-types Y -nclass=1 -maxneg 30
 ```
### Training and testing data are provided in the folder of MusiteDeep/testdata.
