from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class  AddNumbersInput(BaseModel):
    a: int = Field(..., description="The first number to add")
    b: int = Field(..., description="The second number to add")

class AddNumbersTool(BaseTool):
    name: str = "add_numbers"
    description: str = "Add two numbers together."
    args_schema: type[BaseModel] = AddNumbersInput
    def _run(self, a: int, b: int):
        return a + b

    def _arun(self, a: int, b: int):
        return self._run(a, b)
    
tool = AddNumbersTool()
result = tool.invoke({"a": 5, "b": 10})
print(f"Result of AddNumbersTool: {result}")