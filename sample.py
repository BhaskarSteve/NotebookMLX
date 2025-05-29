import torch
from dia.model import Dia

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices."
output = model.generate(text, use_torch_compile=False, verbose=True)

model.save_audio("sample.mp3", output)