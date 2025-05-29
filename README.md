# NotebookMLX

NotebookLM's podcast generation feature for macOS, running locally powered by MLX. Transform any PDF document into an engaging podcast conversation between two AI hosts.

## Installation

### Prerequisites
- uv 
- Python
- macOS with Apple Silicon

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd NotebookMLX
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

## Run

```bash
python main.py --context file_name.pdf --speed 0.95
```

This will:
1. Extract and clean text from the PDF
2. Generate an engaging podcast script
3. Convert the script to natural-sounding audio
4. Save the final audio file to `Output/file_name`

The speed of output is usually fast, set your manual speed accordingly. 

## Customization

### Custom Voice Samples
Create your own custom voice for podcast generation by running:
```bash
python sample.py
```
This will generate a `sample.mp3` file that can be used for voice cloning in your podcast generation. Edit the text in `sample.py` to match the voice characteristics you want to clone.

### Modify Podcast Host Characters
The personalities and characteristics of the podcast hosts (Chris and Sam) can be customized by editing the system prompt in `script.py`. Modify their roles, speaking styles, and interaction patterns to create your desired podcast format.

## Models

- Script generation: `mlx-community/Qwen3-8B-8bit`
- Text cleaning: `mlx-community/Qwen3-1.7B-4bit`
- Text-to-speech: `nari-labs/Dia-1.6B`

## Acknowledgments

- **Qwen Team**: For providing the light weight and excellent open-source Qwen language models. 
- **Nari Labs**: For the outstanding Dia text-to-speech model. Consider contributing to [nari-labs/dia](https://github.com/nari-labs/dia). 
- **Cursor**: For providing the student offer that made this development possible
- **MLX Community**: For porting these models to Apple Silicon and making local AI accessible
