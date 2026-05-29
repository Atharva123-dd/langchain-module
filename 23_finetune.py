# pip install transformers datasets peft trl accelerate bitsandbytes pandas
import os

os.environ["PYTHONUTF8"] = "1"

import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import SFTTrainer


# =========================
# LOAD CSV
# =========================

df = pd.read_csv("dataset.csv")

# print(df.head())


# =========================
# CREATE TRAINING TEXT
# =========================

df.rename(columns={df.columns[2]: "Engine"}, inplace=True)

df["text"] = df.apply(
    lambda row: f"""
### Instruction:
Tell me about the car {row['Cars Names']}.

### Response:
Company: {row['Company Names']}
Engine: {row['Engine']}
Horsepower: {row['HorsePower']}
Top Speed: {row['Total Speed']}
Price: {row['Cars Prices']}
Fuel Type: {row['Fuel Types']}
Seats: {row['Seats']}
Torque: {row['Torque']}
""",
    axis=1
)

# print(df["text"][0])


# =========================
# CONVERT TO DATASET
# =========================

dataset = Dataset.from_pandas(df[["text"]])


# =========================
# LOAD MODEL
# =========================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)


# =========================
# LORA CONFIG
# =========================

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)


# =========================
# TRAINING CONFIG
# =========================

training_args = TrainingArguments(
    output_dir="./car_model",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=1,
    save_strategy="no"
)


# =========================
# TRAINER
# =========================

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)


# =========================
# TRAIN MODEL
# =========================

trainer.train()


# =========================
# SAVE MODEL
# =========================

model.save_pretrained("./car_model")
tokenizer.save_pretrained("./car_model")


print("✅ Fine-tuning complete!")