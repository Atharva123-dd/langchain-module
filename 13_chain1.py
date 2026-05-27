from transformers import pipeline, logging
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser

from pydantic import BaseModel, Field

# -------------------------
# Disable transformers warnings
# -------------------------

logging.set_verbosity_error()

# -------------------------
# Pydantic Model
# -------------------------

class CharacterInfo(BaseModel):
    name: str = Field(..., description="Character name")
    age: int = Field(..., description="Character age")
    city: str = Field(..., description="Character city")

# -------------------------
# Parser
# -------------------------

parser = StrOutputParser()

# -------------------------
# Pipeline
# -------------------------

pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=120,
    do_sample=False,
    temperature=None,
    return_full_text=False,
    clean_up_tokenization_spaces=False,
)

# -------------------------
# LLM
# -------------------------

llm = HuggingFacePipeline(
    pipeline=pipe
)

# -------------------------
# Prompt
# -------------------------

template_1 = PromptTemplate(
    template="""
    GIVE INFO ABOUT THIS TOPIC: {TOPIC}
""",
    input_variables=["TOPIC"],
)

template_2 = PromptTemplate(
    template="""
Write a 5-line summary about the following character: {TEXT}
"""
    ,
    input_variables=["TEXT"]
)
# -------------------------
# Chain
# -------------------------

chain = template_1 | llm | parser  | template_2 | llm

# -------------------------
# Invoke
# -------------------------

result = chain.invoke({
    "TOPIC": "UNEMPLOYEMNT IN INDIA"
})

print(result)
# chain.get_graph().print_ascii()
# print(result.model_dump_json(indent=2))