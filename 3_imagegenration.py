from transformers import pipeline
from diffusers import DiffusionPipeline
import torch

# ----------------------------
# 1. SMALL CHAT MODEL
# ----------------------------

chat_pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=150,
    temperature=0.7,
    return_full_text=False
)

def chat(user_input):
    return chat_pipe(user_input)[0]["generated_text"]


# ----------------------------
# 2. IMAGE MODEL
# ----------------------------

image_pipe = DiffusionPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo",
    torch_dtype=torch.float16
)

device = "cuda" if torch.cuda.is_available() else "cpu"
image_pipe.to(device)


def generate_image(prompt):
    image = image_pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]

    image.save("image.png")
    return "image.png"


# ----------------------------
# 3. SIMPLE CONTROLLER
# ----------------------------

while True:
    user = input("\nAsk (chat/image/exit): ")

    if user == "exit":
        break

    elif user.startswith("image"):
        prompt = user.replace("image", "").strip()
        file = generate_image(prompt)
        print("Image saved:", file)

    else:
        response = chat(user)
        print("\nAI:", response)