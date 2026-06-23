from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tool = StructuredTool.from_function(
    func=add_numbers,
    name="add_numbers",
    description="Add two numbers together.",
    args_schema=AddInput
)

result = tool.invoke({"a": 5, "b": 10})
print(f"Result of StructuredTool: {result}") 