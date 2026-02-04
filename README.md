K-mer Compression-based Anticancer Peptide Classification
This repository contains the implementation of our paper "Compression and k-mer based Approach For Anticancer Peptide Analysis" published in IEEE/ACM Transactions on Computational Biology and Bioinformatics (TCBB).
Overview
Our method introduces a novel compression-based approach for classifying Anti-Cancer Peptides (ACPs) using incremental k-mer encoding and Gzip compression. Unlike traditional methods that compress entire sequences, our approach:

Compresses individual k-mers incrementally to preserve neighboring amino acid context
Uses Normalized Compression Distance (NCD) for pairwise sequence similarity
Generates low-dimensional embeddings via Gaussian kernel and Kernel PCA
Achieves state-of-the-art performance without requiring pre-trained models or extensive hyperparameter tuning

Key Features

Parameter-free: No complex hyperparameter tuning required
Computationally efficient: Alternative to resource-intensive deep neural networks
Generalizable: Effective across multiple cancer types (breast, lung, colon, prostate)
Interpretable: Compression-based distances provide intuitive similarity metrics

Requirements
bashpython>=3.8
numpy>=1.21.5
pandas>=1.3.0
scikit-learn>=1.0.2
nltk>=3.6
Installation

Clone the repository:

bashgit clone https://github.com/sarwanpasha/ACP-Compression-Kmer.git
cd ACP-Compression-Kmer

Install dependencies:

bashpip install -r requirements.txt

Download NLTK data:

pythonimport nltk
nltk.download('punkt')
Dataset
The datasets used in this study are available from:

Breast Cancer ACPs: 949 sequences
Lung Cancer ACPs: 901 sequences
Colon Cancer ACPs: 873 sequences
Prostate Cancer ACPs: 691 sequences

Original dataset source: Grisoni et al., 2019
Dataset Format
Sequences should be stored as NumPy arrays:

{Cancer_Type}_Sequences_{N}.npy: Peptide sequences as strings
{Cancer_Type}_attributes_{N}.npy: Class labels (e.g., "very active", "moderately active", "inactive-experimental", "inactive-virtual")

Place datasets in the data/ directory.
Usage
Basic Usage
pythonfrom compression_kmer import incremental_encoding_distance_matrix, gaussian_kernel
from sklearn.decomposition import KernelPCA
import numpy as np

# Load sequences
sequences = np.load("data/Lungs_Cancer_Sequences_901.npy", allow_pickle=True)

# Compute distance matrix using incremental k-mer compression
distance_matrix = incremental_encoding_distance_matrix(sequences)

# Symmetrize distance matrix
distance_matrix = np.array(distance_matrix)
for i in range(len(distance_matrix)):
    for j in range(len(distance_matrix)):
        if i == j:
            distance_matrix[i,j] = 0
        temp = (distance_matrix[i,j] + distance_matrix[j,i]) / 2
        distance_matrix[i,j] = temp
        distance_matrix[j,i] = temp

# Apply Gaussian kernel
sigma = 1.0
kernel_matrix = gaussian_kernel(distance_matrix, sigma)

# Kernel PCA for dimensionality reduction
transformer = KernelPCA(n_components=500, kernel='precomputed')
embeddings = transformer.fit_transform(kernel_matrix)
Running Complete Pipeline
bashpython run_classification.py --dataset lung --output results/
Arguments

--dataset: Cancer type (breast, lung, colon, prostate)
--data_path: Path to dataset directory (default: data/)
--k_val: K-mer length (default: 3)
--sigma: Gaussian kernel bandwidth (default: 1.0)
--n_components: Number of PCA components (default: 500)
--n_splits: Number of cross-validation splits (default: 5)
--test_size: Test set ratio (default: 0.3)
--output: Output directory for results

Algorithm Details
Core Functions
1. Incremental Encoding Distance Matrix
pythondef incremental_encoding_distance_matrix(sequences, k_val=3):
    """
    Computes pairwise NCD distance matrix using incremental k-mer compression.
    
    Args:
        sequences: List of peptide sequences
        k_val: K-mer length (default: 3)
    
    Returns:
        distance_matrix: n x n matrix of NCD values
    """
```

**Key steps:**
1. Extract k-mers of length `k_val` from each sequence
2. Encode k-mers using count vectorization (tokenize → count vector → string)
3. Incrementally accumulate encoded k-mers
4. Compress using Gzip at each step
5. Compute NCD for all pairs using:
```
   NCD(s1, s2) = (L(s1+s2) - min(L(s1), L(s2))) / max(L(s1), L(s2))
2. Encoding Function
pythondef encode(sequence):
    """
    Converts sequence to numerical representation.
    
    Args:
        sequence: Peptide sequence string
    
    Returns:
        encoded_sequence: String representation of count vector
    """
3. Compression Function
pythondef compress(encoded_sequence):
    """
    Gzip compression of encoded sequence.
    
    Args:
        encoded_sequence: Encoded sequence string
    
    Returns:
        compressed_data: Gzip-compressed bytes
    """
```

## Hyperparameters

Default hyperparameters (optimized through 5-fold cross-validation):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `k_val` | 3 | K-mer length |
| `sigma` | 1.0 | Gaussian kernel bandwidth |
| `n_components` | 500 | Kernel PCA components |
| `test_size` | 0.3 | Train-test split ratio |

### Classifier Settings

- **SVM**: Linear kernel, C=1.0
- **Random Forest**: 100 estimators, max_depth=10
- **MLP**: Hidden layers (10, 10, 10), max_iter=1000
- **KNN**: k=5 neighbors, uniform weights
- **Naive Bayes**: Gaussian
- **Logistic Regression**: liblinear solver

## Results

### Lung Cancer Dataset (901 sequences)

| Method | Accuracy | Precision | Recall | F1 (Weighted) | F1 (Macro) | ROC AUC |
|--------|----------|-----------|--------|---------------|------------|---------|
| **Ours (RF)** | **0.931** | **0.938** | **0.931** | **0.932** | **0.661** | **0.827** |
| ProteinBERT | 0.923 | 0.936 | 0.923 | 0.923 | 0.639 | 0.803 |
| TAPE | 0.913 | 0.920 | 0.913 | 0.912 | 0.655 | 0.802 |
| SeqVec | 0.927 | 0.925 | 0.927 | 0.923 | 0.689 | 0.822 |

### Breast Cancer Dataset (949 sequences)

| Method | Accuracy | Precision | Recall | F1 (Weighted) | F1 (Macro) | ROC AUC |
|--------|----------|-----------|--------|---------------|------------|---------|
| **Ours (RF)** | **0.915** | **0.910** | **0.915** | **0.910** | **0.579** | **0.784** |
| ProteinBERT | 0.893 | 0.893 | 0.893 | 0.893 | 0.602 | 0.779 |
| TAPE | 0.893 | 0.894 | 0.893 | 0.894 | 0.592 | 0.778 |

## Computational Efficiency

Approximate runtime on Intel Xeon E7-4850 v4 @ 2.10GHz with 3TB RAM:

| Dataset Size | Distance Matrix | Kernel PCA | Classification | Total |
|--------------|-----------------|------------|----------------|-------|
| 901 sequences | ~2-3 hours | ~5 min | <1 min | ~2-3 hours |
| 949 sequences | ~2.5-3.5 hours | ~5 min | <1 min | ~2.5-3.5 hours |

**Note**: The distance matrix computation is O(n²k) where n = number of sequences, k = average sequence length. This is a one-time cost; once computed, embeddings can be reused.

## File Structure
```
ACP-Compression-Kmer/
├── README.md
├── requirements.txt
├── compression_kmer.py          # Core implementation
├── run_classification.py        # Full pipeline script
├── data/                        # Dataset directory
│   ├── Breast_Cancer_Sequences_949.npy
│   ├── Breast_Cancer_attributes_949.npy
│   ├── Lungs_Cancer_Sequences_901.npy
│   └── Lungs_Cancer_attributes_901.npy
├── results/                     # Output directory
│   ├── distance_matrices/
│   ├── embeddings/
│   └── classification_results/
└── notebooks/
    └── demo.ipynb              # Jupyter notebook demo
Reproducibility
To reproduce the exact results from the paper:

Use the provided datasets with the same train-test splits:

pythonfrom sklearn.model_selection import ShuffleSplit
sss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

