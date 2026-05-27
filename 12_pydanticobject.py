from transformers import pipeline, logging
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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

parser = PydanticOutputParser(
    pydantic_object=CharacterInfo
)

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
Generate ONE fictional character from {country}.
{format_instructions}

Return ONLY valid JSON.
""",
    input_variables=["country"],
    partial_variables={
        "format_instructions":
        parser.get_format_instructions()
    }
)

template_2 = PromptTemplate(
    template="""Write a 5-line summary about the following character: {character_info}
""",
    input_variables=["character_info"]
)
# -------------------------
# Chain
# -------------------------

chain = template_1 | llm | parser 

# -------------------------
# Invoke
# -------------------------

result = chain.invoke({
    "country": "India"
})

print(result)

print(result.model_dump_json(indent=2))