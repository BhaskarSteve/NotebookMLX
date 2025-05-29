import os
import argparse
import PyPDF2
import time
import gc

import mlx.core as mx
from mlx_lm import load, generate

model_path = "mlx-community/Qwen3-1.7B-8bit"
model, tokenizer = None, None
sys_prompt = """ 
You are text pre-processor. 
I will give you raw text extracted from a PDF. 
Please clean the text and return it as it is without changing anything. 
The raw text sometimes contains formatting issues, citations, latex math, etc. Please remove all these formatting issues and return the text as it is.
Please remove the citations, numbers in between texts and links. Make choices on what to keep and what to remove. 
Rememer and make sure that you are not changing the content in any way, only cleaning it up and presenting in a clean readable format. 
Please do not add markdown formatting or any other special charecters, only return plain text. 
Always start your response directly with processed text, do not ask any questions or acknowledge my request. 
Your output will directly be considered as the final formatted text so make sure to clean and return the text as it is. 
Here is the text: """

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load(model_path)

def clear_model():
    global model, tokenizer
    model, tokenizer = None, None
    mx.clear_cache()
    gc.collect()

def chunk_text_by_words(text, chunk_size=350):
    """Splits text into chunks based on word count."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def process_text_with_llm(text_chunk):
    """Processes a single text chunk using the loaded LLM."""
    conversation = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text_chunk},
    ]

    try:
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        processed_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=2048,
            verbose=False
        )
        return processed_text
    except Exception as e:
        print(f"Error processing chunk with LLM: {e}")
        return text_chunk


def extract_and_process_pdf(pdf_path, word_chunk_size=350, clear: bool = True):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    extracted_text = ""
    try:
        print(f"Opening PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\\n"
                except Exception as e:
                    print(f"Could not extract text from page {i+1}: {e}")

    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during PDF reading: {e}")
        return 
    
    full_text = ""
    chunks = list(chunk_text_by_words(extracted_text, word_chunk_size))
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{total_chunks}...")
        processed_chunk = process_text_with_llm(chunk)
        full_text += processed_chunk + "\n\n"
    
    if clear:
        clear_model()
    return full_text