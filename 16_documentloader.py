from transformers import pipeline, logging
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda


# -------------------------
# Disable warnings
# -------------------------
logging.set_verbosity_error()

# -------------------------
# MODEL (BEST FREE OPTION)
# -------------------------
pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
    clean_up_tokenization_spaces=False,
    top_p=0.9,
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------
# SIMPLE OUTPUT PARSER (NO JSON CRASH)
# -------------------------
parser = StrOutputParser()

# -------------------------
# CLASSIFIER PROMPT
# -------------------------
classifier_prompt = PromptTemplate(
    template="""
You are a sentiment classifier.

Return ONLY one word:
positive or negative

Text:
{TOPIC}
""",
    input_variables=["TOPIC"],
)

classifier_chain = classifier_prompt | llm | parser

# -------------------------
# RESPONSE PROMPTS
# -------------------------
positive_prompt = PromptTemplate(
    template="Write a polite reply to this positive feedback:\n{TOPIC}",
    input_variables=["TOPIC"],
)

negative_prompt = PromptTemplate(
    template="Write an apology for this negative feedback:\n{TOPIC}",
    input_variables=["TOPIC"],
)

# -------------------------
# STEP 1: CLASSIFY
# -------------------------
def classify(x):
    result = classifier_chain.invoke(x).strip().lower()

    if "positive" in result:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return {
        "sentiment": sentiment,
        "TOPIC": x["TOPIC"]
    }

# -------------------------
# STEP 2: BRANCH
# -------------------------
branch_chain = RunnableBranch(
    (lambda x: x["sentiment"] == "positive", positive_prompt | llm | parser),
    (lambda x: x["sentiment"] == "negative", negative_prompt | llm | parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

# -------------------------
# FINAL CHAIN
# -------------------------
chain = RunnableLambda(classify) | branch_chain

# -------------------------
# TEST
# -------------------------
result = chain.invoke({
    "TOPIC": "I hate this product. It is terrible."
})

print("\nFINAL OUTPUT:\n")
print(result)