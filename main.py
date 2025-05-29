import os
import argparse
from pydub import AudioSegment

from tts import tts
from extract import extract_and_process_pdf
from script import generate_podcast_script, convert_to_dialogues

parser = argparse.ArgumentParser()
parser.add_argument("--context", type=str, help="Path to PDF file to extract context from")
parser.add_argument("--script", type=str, help="Path to script file to process")
parser.add_argument("--speed", type=float, default=0.95)
args = parser.parse_args()

clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. [S2] Try it now on Git hub or Hugging Face."
clone_from_audio = "sample.mp3"

if args.context:
    print(f"Processing PDF: {args.context}")
    pdf_path = args.context
    file_name = pdf_path.split("/")[-1].replace(".pdf", "")
    full_text = extract_and_process_pdf(pdf_path)
    print(f"Extracted text, generating script...")
    script, dialogues = generate_podcast_script(full_text)
    print(f"Script and dialogues generated")
    with open(f"{file_name}_script.txt", "w", encoding="utf-8") as f:
        f.write(script)

elif args.script:
    print(f"Processing script: {args.script}")
    script_path = args.script
    file_name = script_path.split("/")[-1].replace(".txt", "")
    with open(script_path, 'r', encoding='utf-8') as f:
        script = f.read()
    dialogues = convert_to_dialogues(script)
    print(f"Dialogues converted")

else:
    print("Error: Please provide either --pdf_path or --script_path")
    exit(1)

for i, dialogue in enumerate(dialogues):
    tts(dialogue, clone_from_text, clone_from_audio, f"Output/dialogue_{i}.wav", True)

combined_audio = AudioSegment.from_wav("Output/dialogue_0.wav")
silence = AudioSegment.silent(duration=400)
for i in range(1, len(dialogues)+1):
    combined_audio += silence
    combined_audio += AudioSegment.from_wav(f"Output/dialogue_{i}.wav")
combined_audio.export(f"Output/{file_name}.wav", format="wav")

for i in range(len(dialogues)):
    os.remove(f"Output/dialogue_{i}.wav")

if args.speed != 1.0:
    audio = AudioSegment.from_wav(f'Output/{file_name}.wav')
    slower_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * args.speed)})
    slower_audio = slower_audio.set_frame_rate(audio.frame_rate)
    slower_audio.export(f'Output/{file_name}.wav', format="wav")

print(f"Final audio saved to: Output/{file_name}.wav")