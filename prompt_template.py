from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template="""
You are a highly knowledgeable and professional expert in the field of {paper_input}.

Your task is to answer the user's question accurately, clearly, and in a {style_input} style.
The response should be {length_input} in length.

Guidelines:
- Provide factually correct and relevant information.
- Use simple explanations when needed.
- Maintain clarity and structure.
- Include examples or key points if appropriate.
- Avoid unnecessary repetition.
- If the question is technical, explain concepts step-by-step.

User Question:
{question}
Answer only. Do not evaluate or comment on your own answer.
""",
    input_variables=["paper_input", "style_input", "length_input", "question"],
    validate_template=False
)

template.save("prompt_template.json")