import torch
import re
from flask import Flask, jsonify
from flask_restful import Resource, Api,reqparse

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)

# --- 1. Define a stopping-criteria to stop when the model emits "<|user|>"
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        for seq in self.stop_ids:
            if input_ids[0].tolist()[-len(seq):] == seq:
                return True
        return False

# --- 2. Model & tokenizer setup
MODEL_PATH = "DeepSeek"  # or HF repo name

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# (Optional) 4‑bit quantization to fit in 8GB VRAM
qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=qconfig,
    device_map="auto"
)
model.eval()

# Prepare stopping criteria for "<|user|>"
stop_token_ids = [
    tokenizer.encode("<|user|>", add_special_tokens=False)
]
stoppers = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# --- 3. Chat history with a single system prompt
history = [
    ("system", "You are a helpful assistant. Answer concisely and do not think out loud.")
]

def build_prompt(history):
    """Convert history into a single prompt string with markers."""
    prompt = ""
    for role, text in history:
        if role == "system":
            prompt += text.strip() + "\n\n"
        elif role == "user":
            prompt += f"<|user|>{text.strip()}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>{text.strip()}\n"
    # indicate model should now reply as assistant
    prompt += "<|assistant|>"
    return prompt

# --- 4. Interactive loop
print("Enter your message (type <reset> to clear history, Ctrl‑C to exit):\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    if user_input.lower() == "<reset>":
        # keep only the system prompt
        history = [history[0]]
        print("[History cleared]\n")
        continue

    # append user message
    history.append(("user", user_input))

    # build & tokenize
    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stoppers,
        do_sample=True,
        top_p=0.95,
        temperature=0.8

    )

    # decode full raw output (with markers)
    raw = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # --- CLEANUP STEPS ---
    # 1) Grab only what comes *after* the last <|assistant|>
    reply_part = raw.split("<|assistant|>")[-1]
    # 2) Discard anything after the next <|user|>
    reply_part = reply_part.split("<|user|>")[0]
    # 3) Remove any <think> tags
    thinkIndex = reply_part.find('</think>')
    if thinkIndex >= 0:
        thinkIndex = thinkIndex + 8
        reply_part = reply_part[thinkIndex:]
    #clean = reply_part.substr(reply_part.indexOf("</think>") + 8)
    clean = re.sub(r"</?think>", "", reply_part)
    clean = re.sub(r"<｜end▁of▁sentence｜>", "", clean)
    # 4) Strip whitespace/newlines
    reply = clean.strip()

    print("Assistant:", reply, "\n")
    history.append(("assistant", reply))
