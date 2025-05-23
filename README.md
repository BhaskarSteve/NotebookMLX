# Podcast Generator

Podcast generator feature of NotebookLM run locally on Mac using MLX. 

### Prerequisites

- Python 3.11+
- 16GB+ VRAM
- MLX
- uv

### Usage

```bash
python podcast_generator.py --input context.txt
```

### Command Line Options

- `--input, -i`: Path to biography text file
- `--output, -o`: Output directory for generated files

<!-- ## 📁 Output Files

The generator creates three files in the output directory:

1. **`podcast_script_raw.txt`**: Original script with proper formatting and speaker labels
2. **`podcast_script_formatted.txt`**: Script formatted for TTS with `[S1]` and `[S2]` tags
3. **`podcast_final.wav`**: Final audio podcast (44.1kHz WAV format) -->

<!-- ## 🎯 How It Works

### 1. Script Generation
- Uses **Qwen3-8B-4bit** language model to create engaging dialogue
- Two distinct hosts: Alex (curious, business-focused) and Sam (analytical, contextual)
- Generates 15-20 minute conversations (~2000-2500 words)
- Natural speech patterns with reactions and follow-up questions

### 2. TTS Conversion
- Formats script for **Dia-1.6B** TTS model using `[S1]` and `[S2]` speaker tags
- Converts natural speech elements (laughs, pauses, etc.)
- Generates high-quality audio with distinct voices for each speaker

### 3. Quality Optimizations
- Temperature-controlled generation for consistent quality
- Proper chat template handling for better model performance
- Memory-efficient model loading for 16GB systems -->

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
