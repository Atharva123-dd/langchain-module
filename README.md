# 🤖 LangChain + HuggingFace Chat Agent

A simple AI chatbot built using LangChain, HuggingFace Transformers, and optional Streamlit UI.

---

# 🚀 Features

- Local LLM (HuggingFace Transformers)
- CLI + Streamlit chatbot
- Chat history memory
- LangChain prompts
- Lightweight models (TinyLlama / Qwen)

---

# 📦 Install

pip install transformers torch accelerate langchain langchain-huggingface streamlit

---

# ⚙️ Setup

Create venv:
py -3.12 -m venv venv

Activate:
venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

---

# ▶️ Run

CLI:
python app.py

Streamlit:
streamlit run app.py

---

# 🧠 Models

- Qwen/Qwen2.5-0.5B-Instruct
- TinyLlama/TinyLlama-1.1B-Chat-v1.0

---

# 💬 Example

You: What is AI?
Bot: AI stands for Artificial Intelligence

---

# 🧩 Structure

langchain/
├── app.py
├── requirements.txt
├── prompt_template.json
├── venv/
└── README.md

---

# ⚠️ Fix Issues

pip upgrade:
python -m pip install --upgrade pip

venv activate:
venv\Scripts\activate

---

# 🚀 Next Steps

- Google Search Agent
- Wikipedia Tool
- Memory (vector DB)
- Real AI Agent

---

# 👨‍💻 Author

Built while learning LangChain + HuggingFace