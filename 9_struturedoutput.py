from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import TypedDict
import json
import re

def extract_summary_sentiment(text):
    # find all JSON blocks
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for m in reversed(matches):  # take last valid one
        try:
            data = json.loads(m)
            if "summary" in data and "sentiment" in data:
                return {
                    "summary": data["summary"],
                    "sentiment": data["sentiment"]
                }
        except:
            continue

    return {
        "summary": "not found",
        "sentiment": "unknown"
    }
    

class Response(TypedDict):
    summary: str
    sentiment: str

parser = JsonOutputParser(pydantic_object=Response)

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=120,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a STRICT JSON generator.

RULES:
- Output ONLY JSON
- No explanation
- No extra text
- No markdown

FORMAT:
{{
  "summary": "...",
  "sentiment": "positive"
}}

IMPORTANT:
Response must start with {{ and end with }}.
"""),
    ("human", "{input}")
])

chain = prompt | llm | parser

user_input = input("Enter your text: ")
response = chain.invoke({
    "input": user_input
})

raw_output = response  # your LLM output string

result = extract_summary_sentiment(raw_output)

print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])