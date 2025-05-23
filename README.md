# Podcast Generator

Podcast generator feature of NotebookLM run locally on Mac using MLX. 

### Prerequisites

- Python 3.11+
- 16GB+ VRAM
- MLX
- uv

### Usage

```bash
python main.py --input context.txt
```

### Command Line Options

- `--input, -i`: Path to biography text file
- `--output, -o`: Output directory for generated files

## Technical Details

### Models Used
- **Language Model**: `mlx-community/Qwen3-8B-4bit` (MLX optimized)
- **TTS Model**: `nari-labs/Dia-1.6B` (PyTorch)

### Dependencies
- `mlx-lm`: For running Qwen3 model on Apple Silicon
- `dia`: For text-to-speech conversion
- `soundfile`: For audio file handling
- `torch`: Required by Dia model

## Acknowledgments

- [Nari Labs](https://github.com/nari-labs/dia) for the Dia TTS model
- [MLX Community](https://huggingface.co/mlx-community) for Qwen3 MLX conversion
- [Alibaba](https://huggingface.co/Qwen) for the original Qwen models
