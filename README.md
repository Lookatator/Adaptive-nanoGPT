# Adaptive Attention Transformer

This repository is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT).

## Overview

This project implements a novel technique to reduce the computational cost of attention layers in transformer models. Our approach introduces an adaptive learnable periodic masking mechanism that selectively masks queries during attention weight computation. When combined with [Adaptive Attention Span (Sukhbaatar et al., 2019)](https://arxiv.org/abs/1905.07799v2), our method significantly improves computational efficiency by reducing floating point operations (FLOPs) in attention calculations while maintaining model performance.

## Key Features

- Novel adaptive periodic masking mechanism for selective query masking
- Integration with Adaptive Attention Span
- Significant reduction in attention computation FLOPs.

## Installation

```python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

For enwik8 dataset:
```bash
bash get_data.sh
cd data/enwik8
python prep_enwik8.py
```

## Usage

### Training

Basic training command with adaptive periodic masking and attention span:

```bash
python train.py config/train_enwik8.py --use_adaptive_attention=True --use_triangle_wave_masking=True --compile=True 
```

### Key Parameters

- `use_adaptive_attention`: Enable adaptive attention span mechanism (Sukhbaatar et al., 2019)
- `use_triangle_wave_masking`: Enable adaptive periodic masking (our approach), requires `use_adaptive_attention` to be set to true.
- `softness_span_mask`: Softness parameter for the masking function (default: 32)
- `span_reg`: L1 regularization coefficient for span parameters (default: 4e-6)
- `period_min_triangle_wave_masking`: Minimum period for triangle wave (default: 2.0)
- `period_max_triangle_wave_masking`: Maximum period for triangle wave (default: 8.0)

### Available Configurations

The repository includes a pre-made configuration for the enwik8 dataset in `config/train_enwik8.py`.

## Model Architecture

The adaptive attention mechanism and triangle wave masking are implemented in `adaptive_attention_span.py`. This file contains the `AdaptiveCausalAttention` class which extends the base attention mechanism with adaptive span and periodic masking functionality.


