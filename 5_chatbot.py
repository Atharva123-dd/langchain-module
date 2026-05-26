from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer
)
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

chat_history = []
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# Streamer
streamer = TextStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

while True:

    user_input = input("\nYou: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": chat_history
        }
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")

    print("\nAI: ", end="")

    # Generate live
    model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.3,
        do_sample=True,
        streamer=streamer
    )

    print()