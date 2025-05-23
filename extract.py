import os
import argparse
import PyPDF2
import time # Add time for potential timing/debugging

from mlx_lm import load, generate

# Define Model Path and System Prompt
model_path = "mlx-community/Qwen3-4B-4bit"
# SYS_PROMPT = """You are an expert text processing assistant. Your task is to take the following text extracted from a PDF and clean it up. \
# Remove extraneous line breaks, fix spacing issues, correct obvious OCR errors if possible, and format it into coherent paragraphs. \
# Ensure the flow between sentences and paragraphs is natural. Do not add any commentary, preamble, or explanation, just output the cleaned text."""

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

# Load Model and Tokenizer (Load once)
print(f"Loading model: {model_path}...")
model, tokenizer = load(model_path)
print("Model loaded successfully.")


def chunk_text_by_words(text, chunk_size=350):
    """Splits text into chunks based on word count."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])


def process_text_with_llm(text_chunk):
    """Processes a single text chunk using the loaded LLM."""
    conversation = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text_chunk},
    ]

    try:
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # Estimate prompt tokens roughly (optional, for debugging)
        # prompt_tokens = len(tokenizer.encode(prompt))
        # print(f"Prompt tokens (estimated): {prompt_tokens}")

        processed_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=1024, # Increased max_tokens to accommodate potentially longer cleaned outputs
            verbose=False # Keep verbose off for cleaner output unless debugging
        )
        # Add a small delay if hitting rate limits or for stability, though unlikely with local models
        # time.sleep(0.1)
        return processed_text
    except Exception as e:
        print(f"Error processing chunk with LLM: {e}")
        # Return original chunk or empty string on error? Returning original for now.
        return text_chunk


def extract_and_process_pdf(pdf_path, word_chunk_size=350):
    """
    Extracts text from a PDF, preprocesses it using an LLM chunk by chunk,
    and saves the result to a .txt file.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    # Create Examples directory if it doesn't exist
    output_dir = "Examples"
    os.makedirs(output_dir, exist_ok=True)

    # Determine output file path
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_txt_path = os.path.join(output_dir, f"{base_name}.txt") # Changed suffix

    extracted_text = ""
    try:
        print(f"Opening PDF: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            print(f"Reading {num_pages} pages...")

            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\\n"
                except Exception as e:
                    print(f"Could not extract text from page {i+1}: {e}")
            print(f"Finished extracting text (approx {len(extracted_text.split())} words).")

    except PyPDF2.errors.PdfReadError as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return # Stop if PDF reading fails
    except Exception as e:
        print(f"An unexpected error occurred during PDF reading: {e}")
        return # Stop on other errors during reading

    # Process the extracted text chunk by chunk
    print("Starting LLM processing...")
    processed_full_text = ""
    chunks = list(chunk_text_by_words(extracted_text, word_chunk_size))
    total_chunks = len(chunks)
    start_time = time.time()

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{total_chunks}...")
        processed_chunk = process_text_with_llm(chunk)
        processed_full_text += processed_chunk + "\n\n" # Add space between processed chunks

    end_time = time.time()
    print(f"LLM processing finished in {end_time - start_time:.2f} seconds.")

    # Save the processed text
    print(f"Saving processed text to {output_txt_path}...")
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as outfile:
            outfile.write(processed_full_text.strip()) # Use strip to remove trailing newlines
        print("Processed text saved successfully.")
    except Exception as e:
        print(f"Error writing output file {output_txt_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract text from a PDF, process it with an LLM, and save as .txt.')
    parser.add_argument('pdf_path', type=str, help='Path to the input PDF file')
    parser.add_argument('--chunk_size', type=int, default=350, help='Number of words per chunk for LLM processing')

    args = parser.parse_args()

    extract_and_process_pdf(args.pdf_path, args.chunk_size)
