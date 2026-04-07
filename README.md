# TinyStories-LM: Tiny Language Models on TinyStories

A rigorous PyTorch implementation of decoder-only transformer language models trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (Eldan & Li, 2023). This codebase provides a systematic framework for investigating scaling properties, architectural choices, and training dynamics of compact language models across three parameter regimes: 9M, 19M, and 40M parameters.

## Quick Start

### Inference & Generation

The trained 40M weights can be obtained from here: [Trained Model](https://drive.google.com/drive/folders/12hX7t6NFgQI5pfisatJ7rvMCQWtZeBrm?usp=sharing)

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
├── config/
│   ├── lm_v1_9M.py          # 9M
│   ├── lm_v1_19M.py         # 19M
│   ├── lm_v1_40M.py         # 40M
│   ├── dataset.py           # Dataset config
│   ├── training.py          # Training config
│   └── utils.py
├── src/
│   ├── train.py             # Training loop
│   ├── corpus.py            # Pretraining dataset
│   ├── instruct.py          # Fine-tuning dataset
│   ├── generation.py        # Text generation utils
│   ├── tokenizer.py         # Tokenizer
│   ├── serialization.py     # Model checkpointing
│   ├── utils.py             # utilities
│   └── modules/             # Model architecture
│       ├── blocks.py
│       ├── decoder.py
│       ├── rope.py
│       ├── moe.py
│       └── utils.py
├── notebooks/                # Jupyter notebooks
│   ├── causal_guide.ipynb
│   ├── exploratory.ipynb
│   └── model_info.ipynb
├── script.py                 # Main training script
├── requirements.txt          # Python dependencies
└── README.md
```

## Model Architecture

### Decoder Block

```
Input → RMSNorm → Multi-Head Attention → Residual → RMSNorm → SwiGLU FFN → Residual → Output
```

**Key Components**:
- **Multi-Head Attention**: self-attention with RoPE positional embeddings
- **RMSNorm**: RMS-based layer normalization
- **SwiGLU**: Gated linear unit activation function
- **Residual Connections**: Pre-norm residual connections for stable training
- **Weight Tying**: Shares weights between token embedding and output projection layers
- **MoE (Mixture of Experts)**: Optional sparse layer for parameter efficiency. Dynamically routes tokens to the most relevant experts

### Full Decoder Architecture

```
Input → Embedding → N Decoder Blocks → RMSNorm → Projection → Output
```

### Model Specifications

| Parameter | 9M | 19M | 40M |
|-----------|-----|-----|-----|
| Embedding Dim (d_model) | 128 | 256 | 512 |
| Num Heads (h) | 8 | 8 | 8 |
| FFN Dim (d_ff) | 512 | 1024 | 2048 |
| Num Layers | 8 | 6 | 8 |
| Dropout | 0.1 | 0.1 | 0.1 |

NOTE: the number of parameter might be a little inaccurate

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

Contributions are welcome, Feel free to open issues or submit pull requests for improvements.
