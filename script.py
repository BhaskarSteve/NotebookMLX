import os
import re
import time
import argparse
import gc
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model_path = "mlx-community/Qwen3-8B-4bit"
model, tokenizer = None, None
sys_prompt = """
# ROLE & OBJECTIVE
You are an expert podcast scriptwriter specializing in creating engaging conversational content. Your task is to transform the provided context into a compelling 20-30 minute podcast script featuring two hosts with natural chemistry.

# CHARACTER PROFILES
**Chris**: The curious questioner who drives discovery
- Asks thought-provoking questions
- Provides analogies and insights
- Focuses on "why" and "wow" moments

**Sam**: The knowledgeable explainer who delivers answers
- Provides detailed explanations
- Uncovers surprising facts
- Focuses on "how" and "what" details

# SCRIPT STRUCTURE REQUIREMENTS

## Opening (First 30 seconds)
- Start with an immediate hook that grabs attention
- NO introductions or setup - dive straight into content
- Begin with either Chris or Sam speaking

## Body (Main content)
- Build momentum through escalating discoveries
- Include 2-3 "wow factor" moments
- Create natural conversation flow with:
  - Genuine surprise and excitement
  - Questions that lead to deeper exploration
  - Moments where hosts learn from each other
  - Building tension and reveals

## Closing (Final 30 seconds)
- End with a powerful insight or thought-provoking question
- Leave audience wanting more

# DIALOGUE FORMATTING RULES

## Required Format
Each line must follow this exact pattern:
**Chris**: [Clean dialogue without verbal cues] (Optional verbal cues at the end)
**Sam**: [Clean dialogue without verbal cues] (Optional verbal cues at the end)

## Verbal Cues Policy
- DEFAULT: Use NO verbal cues in 90%+ of dialogue lines
- EXCEPTION: Only add verbal cues when absolutely essential for meaning
- PLACEMENT: If used, place ONLY at the end of dialogue: "That's incredible (laughs)"
- NEVER start dialogue with verbal cues
- Available cues: (laughs), (sighs), (gasps)
- Use maximum of 2-3 verbal cues in the entire script

## What NOT to include
- No background music references
- No sound effects
- No stage directions
- No narrator text
- No introductory explanations

# CONTENT APPROACH
Transform the provided context into engaging dialogue that reveals surprising connections, challenges assumptions, and creates genuine discovery moments between the hosts.

# CONTEXT TO TRANSFORM
--- START OF CONTEXT ---
{context_text}
--- END OF CONTEXT ---

Generate the podcast script now, starting immediately with the first speaker:
"""

def load_model():
    """Loads the LLM model and tokenizer."""
    global model, tokenizer
    if model is None or tokenizer is None:
        model, tokenizer = load(model_path)

def generate_podcast_script(context_text: str, clear: bool = True) -> str:
    """
    Generates a podcast script using the loaded LLM based on the provided context.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        load_model()

    formatted_prompt = sys_prompt.format(context_text=context_text)
    conversation = [
        {"role": "user", "content": formatted_prompt}
    ]

    try:
        prompt_for_model = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )        
        sampler = make_sampler(temp=0.7, top_p=0.9, min_p=0.0, min_tokens_to_keep=1)
        generated_script = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_for_model,
            max_tokens=4096,
            sampler=sampler,
            verbose=False
        )
        
        cleaned_script = re.sub(r'<think>.*?</think>', '', generated_script, flags=re.DOTALL)
        cleaned_script = cleaned_script.replace('*', '')
        
        if clear:
            model, tokenizer = None, None
            mx.clear_cache()
            gc.collect()

        dialogues = convert_to_dialogues(cleaned_script)
        return cleaned_script, dialogues

    except Exception as e:
        print(f"Error generating script with LLM: {e}")
        if clear:
            model, tokenizer = None, None
            mx.clear_cache()
            gc.collect()
        return "Error: Could not generate podcast script."

def convert_to_dialogues(cleaned_script: str):
    lines = cleaned_script.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if line:
            line = line.replace('Chris:', '[S1] ')
            line = line.replace('Sam:', '[S2] ')
            processed_lines.append(line)

    paired_lines = []
    for i in range(0, len(processed_lines) - 1, 2):
        if i + 1 < len(processed_lines):
            paired_line = f"{processed_lines[i]} {processed_lines[i + 1]}"
            paired_lines.append(paired_line)
        else:
            paired_lines.append(processed_lines[i])
    return paired_lines