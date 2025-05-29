import torch
from dia.model import Dia

def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Dia.from_pretrained(
        "nari-labs/Dia-1.6B", 
        compute_dtype="float16",
        device=device
    )
    return model

def tts(text: str, clone_from_text: str = None, clone_from_audio: str = None, output_path: str = None, clear: bool = True) -> str:
    model = load_model()
    if clone_from_text and clone_from_audio:
        output = model.generate(clone_from_text + text, audio_prompt=clone_from_audio, use_torch_compile=False, verbose=False)
    else:
        output = model.generate(text, use_torch_compile=False, verbose=False)

    model.save_audio(output_path, output)
    
    if clear:
        del model
        model = None
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return output_path