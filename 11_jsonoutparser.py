# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

# parser = JsonOutputParser()

# pipe = pipeline(
#     "text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     max_new_tokens=120,
#     do_sample=False,
#     return_full_text=False,
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # ✅ FIXED templates
# template_1 = PromptTemplate(
#     template='Give me name , age and city if a fictional character \n {format_instructions}',
#     input_variables=[],
#     partial_variables={"format_instructions": parser.get_format_instructions()}
# )


# prompt = template_1.format()

# result = llm.invoke(prompt)

# print(result)



from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=120,
    do_sample=False,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ FIXED TEMPLATE
template_1 = PromptTemplate(
    template="""
Give me name, age, and city of a fictional character.

{format_instructions}

Return ONLY valid JSON.
""",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# ❌ DO NOT .format()
prompt = template_1.format()

# ⚠️ Just LLM call (no parser yet because output is messy for TinyLlama)
result = llm.invoke(prompt)

print(result)