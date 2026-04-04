# TinyStories-LM: Tiny Language Models on TinyStories

A rigorous PyTorch implementation of decoder-only transformer language models trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (Eldan & Li, 2023). This codebase provides a systematic framework for investigating scaling properties, architectural choices, and training dynamics of compact language models across three parameter regimes: 9M, 19M, and 40M parameters.

## Quick Start

### Inference & Generation

See [Test and Result.ipynb](Test%20and%20Result.ipynb) for interactive examples:

- **Next Word Prediction**: Given a prompt, generate the next tokens using the pretrained model
- **Story Generation**: Fine-tuned model generates complete stories based on features, keywords, and summary

```python
from src.generation import ModelLoader, generate_story

config = load_config(model_40M_params)
loader = ModelLoader(config, stage="finetuning", version="v1_40M", load_from_epoch=1)

loader.autoregressive_attribute(
    max_new_tokens=2000,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
)

features = ["Dialogue", "BadEnding"]
words = ["ride", "upset"]
summary = "Lily steals a bike and gets caught."
story = generate_story(loader, features, words, summary)
```

## Project Structure

```
TinyLM/
├── config/                           # Configuration modules
│   ├── lm_v1_9M.py                  # 9M parameter model config
│   ├── lm_v1_19M.py                 # 19M parameter model config
│   ├── lm_v1_40M.py                 # 40M parameter model config
│   ├── dataset.py                   # Dataset configurations
│   ├── training.py                  # Training configurations
│   └── utils.py                     # Config loading utilities
├── src/                              # Source code
│   ├── train.py                     # Training loop implementation
│   ├── corpus.py                    # Pretraining dataset handling
│   ├── instruct.py                  # Fine-tuning dataset handling
│   ├── generation.py                # Text generation utilities
│   ├── tokenizer.py                 # Tokenizer training and loading
│   ├── serialization.py             # Model checkpointing
│   ├── callbacks.py                 # Training callbacks
│   ├── plot.py                      # Training visualization
│   ├── utils.py                     # Helper utilities
│   └── modules/                     # Model architecture
│       ├── blocks.py                # Core building blocks (attention, norm, embedding)
│       ├── decoder.py               # Decoder-only transformer
│       ├── rope.py                  # Rotary positional embeddings
│       ├── moe.py                   # Mixture of Experts layer
│       └── utils.py                 # Module utilities
├── notebooks/                        # Jupyter notebooks
│   ├── causal_guide.ipynb          # Causality attention mask explanation
│   ├── exploratory.ipynb           # Data exploration
│   └── model_info.ipynb            # Model architecture details
├── script.py                         # Main training script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Architecture

### Decoder Block

```
Input → RMSNorm → Multi-Head Attention → Residual → RMSNorm → SwiGLU FFN → Residual → Output
```

**Key Components**:
- **Multi-Head Attention**: 8-16 attention heads with RoPE positional embeddings
- **RMSNorm**: RMS-based layer normalization (more efficient than LayerNorm)
- **SwiGLU**: Gated linear unit activation function
- **Residual Connections**: Pre-norm residual connections for stable training

### Model Specifications

| Parameter | 9M | 19M | 40M |
|-----------|-----|-----|-----|
| Vocab Size | 10,000+ | 10,000+ | 10,000+ |
| Embedding Dim (d_model) | 384 | 512 | 768 |
| Num Heads (h) | 8 | 8 | 16 |
| FFN Dim (d_ff) | 1024 | 1536 | 2560 |
| Num Layers | 6 | 9 | 12 |
| Dropout | 0.1 | 0.1 | 0.1 |

## Training Pipeline

### Pretraining Stage

**Objective**: Causal language modeling on raw TinyStories corpus

- **Dataset**: HuggingFace TinyStories Corpus

### Fine-tuning Stage

**Objective**: Instruction-following with structured conditioning

- **Conditioning Format**:
  ```
  [FEATURES_START] <features> [FEATURES_END]
  [WORDS_START] <keywords> [WORDS_END]
  [SUMMARY_START] <summary> [SUMMARY_END]
  [STORY_START] <story> [STORY_END]
  ```

- **Dataset**: HuggingFace TinyStories Instruct

## Configuration

Model and training configs are defined in `config/` directory.


## Text Generation Strategies

### Temperature Sampling
Controls randomness of predictions. Higher values = more random.
```python
temperature=0.7  # Default: 1.0
```

### Top-k Filtering
Only sample from the k most likely next tokens.
```python
top_k=50  # Sample from top 50 tokens
```

### Nucleus (Top-p) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds p.
```python
top_p=0.95  # Cumulative probability threshold
```

## Results

Training results and generation samples are logged in [Test and Result.ipynb](Test%20and%20Result.ipynb):

- Next-word prediction examples
- Story generation with various prompts and parameters

## Command Reference

### Training Arguments

```
--model_size {9M, 19M, 40M}    Model size variant (required)
--version TEXT                 Version identifier for checkpoint (required)
--phase {pretraining, finetuning}  Training phase (required)
--state {initial, resume}      Whether to start fresh or resume (required)
--load_epoch INT               Epoch to load from (required for resume)
--test                         Run in test mode with small datasets
```

## Key Features Explained

### RoPE (Rotary Positional Embeddings)
Instead of adding position encodings to embeddings, RoPE rotates query and key vectors based on position. More efficient and generalizes better to longer sequences.

### SwiGLU Feed-Forward Network (FFN)
An improved variant of the standard FFN that uses a gated activation mechanism. Instead of a single activation (like ReLU or GELU), SwiGLU splits the input into two parts: one is activated with a Swish function, and the other acts as a gate that modulates the output.

### Mixture of Experts (MoE)
Optional sparse layer for parameter efficiency. Dynamically routes tokens to the most relevant experts, reducing computation while maintaining capacity.

### Weight Tying
Shares weights between token embedding and output projection layers, reducing parameters without sacrificing expressiveness.

## Citation

Based on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset:

```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronan and Li, Yonatan},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Author

@2025-2026 Muhammad Nizwa. All rights reserved.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.
