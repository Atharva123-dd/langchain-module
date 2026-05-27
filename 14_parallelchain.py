from transformers import pipeline, logging
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel

# -------------------------
# Disable transformers warnings
# -------------------------

logging.set_verbosity_error()


# -------------------------
# Parser
# -------------------------

parser = StrOutputParser()

# -------------------------
# Pipeline
# -------------------------
# low specification models like TinyLlama-1.1B-Chat-v1.0 can be used for testing, but for better results, consider using more powerful models like Mistral-7B-Instruct-v0.2 or similar.
# pipe = pipeline(
#     task="text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     max_new_tokens=500,
#     do_sample=True,
#     temperature=0.7,
#     return_full_text=False,
#     clean_up_tokenization_spaces=False,
#     top_p=0.9,
# )

# highspecification models like Mistral-7B-Instruct-v0.2 or similar can be used for better results, but for testing, you can use low specification models like TinyLlama-1.1B-Chat-v1.0.
pipe = pipeline(
    task="text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False,
)

# -------------------------
# LLM
# -------------------------

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------
# Prompt
# -------------------------

template_1 = PromptTemplate(
    template="""
    Write bullet-point study notes ONLY about the given topic. Do not add unrelated content. \n  {TOPIC}
""",
    input_variables=["TOPIC"],
)

template_2 = PromptTemplate(
    template="""
    Generate 5 simple questions about \n {TOPIC}
""",
    input_variables=["TOPIC"],
)


template_3 = PromptTemplate(
    template="""
    merge the following notes and question into a single text: \n
    -> notes:{notes} \n
    -> question:{question}
""",
    input_variables=["notes", "question"],
)
# -------------------------
# parallelChain
# -------------------------

parallel_chain = RunnableParallel(
    {"notes": template_1 | llm | parser, "question": template_2 | llm | parser}
)

merge_chain = template_3 | llm | parser

chain = parallel_chain | merge_chain
# -------------------------
# Invoke
# -------------------------

result = chain.invoke(
    {
        "TOPIC": "SQL stands for Structured Query Language. It is the standard programming language used to communicate, manage, and manipulate data within a relational database. Think of it as a specialized language that lets you ask questions and give instructions to a database so you can find exactly what you need.Main Types of SQL CommandsSQL is broken down into sub-languages based on functionality:Data Query Language (DQL): Used to fetch data (e.g., SELECT).Data Manipulation Language (DML): Used to add, change, or erase records (e.g., INSERT, UPDATE, DELETE).Data Definition Language (DDL): Used to design or alter table structures (e.g., CREATE, ALTER, DROP).Data Control Language (DCL): Used to manage user permissions and security (e.g., GRANT, REVOKE).Key BenefitsUniversal Standard: It is an official standard recognized by ANSI and ISO.English-Like Syntax: It reads similarly to English sentences, making it easier to learn.Highly Efficient: It can search through millions of rows of data within seconds.Declarative Approach: You only state what data you want, and the system figures out how to get it.Popular SQL Database SystemsWhile SQL is the underlying language, various software platforms implement it:MySQL (Open-source, widely used for web development)PostgreSQL (Advanced, highly customizable open-source system)Microsoft SQL Server (Enterprise database platform by Microsoft)Oracle Database (Robust system used by major corporations)"
    }
)

print(result)
# parallel_chain.get_graph().print_ascii()
# print(result.model_dump_json(indent=2))
