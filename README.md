# DeepVarPred

DeepVarPred is a deep learning tool that predicts the pathogenicity of gene variants using image-based representation of genomic data. The tool converts genomic sequences and variant information into RGB images and uses a ResNet34-based model to classify variants as pathogenic or benign.

## Overview

DeepVarPred processes genetic mutation data through the following steps:
1. Converts mutation information into RGB images by encoding:
   - Reference sequence (Red channel)
   - Variant sequence (Green channel)
   - Conservation scores (Blue channel)
2. Trains a ResNet34 model on these images to classify variants
3. Evaluates model performance and exports to ONNX format for deployment

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Output Files](#output-files)
- [Performance Metrics](#performance-metrics)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended)
- 16GB+ RAM

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/DeepVarPred.git
cd DeepVarPred
```

### 2. Create a virtual environment (recommended)

Using conda:
```bash
conda create -n deepvarpred python=3.8
conda activate deepvarpred
```

Or using venv:
```bash
python -m venv deepvarpred-env
# On Windows
deepvarpred-env\Scripts\activate
# On Linux/Mac
source deepvarpred-env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
scikit-learn~=1.2.0
pandas~=2.0.3
numpy~=1.23.4
torch~=1.13.1
torchvision~=0.14.1
onnxruntime~=1.13.1
pillow~=9.3.0
tqdm~=4.64.1
biopython~=1.80
opencv-python~=4.6.0
```
## Quick Start
To run DeepVarPred for a specific gene, follow these steps:
```bash
cd PKD1
```
Then, run the following command:
```bash
python DeepVarPred.py --gene PKD1 --batch_size 32 --epochs 10
```

This will:
1. Process the variant files
2. Create image representations
3. Split into training, validation, and test sets
4. Train a ResNet34 model
5. Evaluate performance and save models
6. Export the trained model to ONNX format


## Data Preparation for Your Gene

DeepVarPred requires specific input files organized in a particular structure:

### Required Input Files

1. Create a `ref` directory in your project root:
```bash
mkdir -p ref
```

2. Add the following files to the `ref` directory:
   - `GENE_p.txt`: Tab-separated file containing pathogenic variants
   - `GENE_b.txt`: Tab-separated file containing benign variants
   - `GENE_ref.fa`: FASTA file with the reference sequence
   - `GENE_.bed`: BED file with gene coordinates

Where `GENE` is the name of your gene of interest (e.g., BRCA2).

### Format of Input Files

#### Variant files (`GENE_p.txt` and `GENE_b.txt`)

Tab-separated files with the following columns:
- `#chr`: Chromosome
- `pos(1-based)`: Variant position
- `ref`: Reference allele
- `alt`: Alternate allele
- `hg19_chr`: hg19 chromosome
- `hg19_pos(1-based)`: hg19 position
- `hg18_chr`: hg18 chromosome
- `hg18_pos(1-based)`: hg18 position
- `clinvar_clnsig`: Clinical significance
- Multiple conservation score columns

Example:
```
#chr    pos(1-based)    ref    alt    hg19_chr    hg19_pos(1-based)    hg18_chr    hg18_pos(1-based)    clinvar_clnsig    score1    score2    ...
13      32890572        G      A      13          32890572             13          31789722             Pathogenic        0.9876    0.8765    ...
```

#### Reference sequence (`GENE_ref.fa`)

Standard FASTA format:
```
>BRCA2 reference sequence
ATGGCTTCGAAATTAAAAAGTCTTCTAACTTCTGAAACAGACTTCGAAATTTTTTTTTA
GAATCTGCTTGTTTCAAGTCAGCTCCTTTTGAAGGTGAGAAAAATGATAATGATCTTTC
...
```

#### Gene coordinates (`GENE_.bed`)

BED format with at least 3 columns:
```
chr13   32889611    32973347    BRCA2
```

## Usage

### Basic Usage

To run DeepVarPred with default parameters:

```bash
python DeepVarPred.py --gene BRCA2
```

This will:
1. Process the variant files
2. Create image representations
3. Split into training, validation, and test sets
4. Train a ResNet34 model
5. Evaluate performance and save models

### Advanced Options

```bash
python DeepVarPred.py --gene BRCA2 --batch_size 64 --epochs 20
```

Parameters:
- `--gene`: Name of the gene (default: BRCA2)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 10)

## Model Architecture

DeepVarPred uses a ResNet34 architecture pre-trained on ImageNet and fine-tuned for variant classification. The model:

1. Takes RGB images (300×300) as input
2. Processes them through the ResNet34 convolutional layers
3. Outputs a binary classification (pathogenic/benign)

## Output Files

After running, DeepVarPred creates:

- `dataset/`: Directory containing processed images
  - `train/p/`: Training pathogenic images
  - `train/b/`: Training benign images
  - `val/p/`: Validation pathogenic images
  - `val/b/`: Validation benign images
  - `test/p/`: Testing pathogenic images
  - `test/b/`: Testing benign images

- `model/`: Directory containing trained models
  - `all_resnet34_300.pth`: PyTorch model weights
  - `all_resnet34_300.onnx`: ONNX format model for deployment

## Performance Metrics

DeepVarPred reports several metrics to evaluate model performance:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

The metrics are printed in the console output after training and evaluation.

## Customization

### Using a Different Gene

To use DeepVarPred with a different gene:

1. Prepare the required input files (`GENE_p.txt`, `GENE_b.txt`, `GENE_ref.fa`, `GENE_.bed`)
2. Run the script with the `--gene` parameter:
```bash
python DeepVarPred.py --gene YOUR_GENE_NAME
```

### Hyperparameter Tuning

You can adjust training parameters:
```bash
python DeepVarPred.py --gene BRCA2 --batch_size 64 --epochs 20
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Try reducing batch size: `--batch_size 16`
   - If issues persist, check GPU memory usage with `nvidia-smi`

2. **Missing input files**
   - Ensure all required files exist in the `ref` directory
   - Check file naming: they should match the gene name format

3. **ImportError: No module named 'X'**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check your Python environment is activated

4. **ValueError: Empty input for conservation score**
   - Verify your variant files have the correct format and contain conservation scores

### Getting Help

If you encounter issues not covered here, please open an issue on the GitHub repository with:
- Error message and stack trace
- Command used to run the script
- Description of input data
- System specifications

## License

This project is licensed under the MIT License - see the LICENSE file for details.