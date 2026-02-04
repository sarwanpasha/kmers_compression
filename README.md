# K-mer Compression-based Anticancer Peptide Classification

This repository contains the implementation of our paper "Compression and k-mer based Approach For Anticancer Peptide Analysis".

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
