from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=120,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ FIXED templates
template_1 = PromptTemplate.from_template(
    "Write a detailed explanation about: {input}"
)

template_2 = PromptTemplate.from_template(
    "Write a 5-line summary about: {input}"
)

chain= template_1 | llm | parser | template_2 | llm | parser

result = chain.invoke({
    "input": "earth and moon"
})


print(result)