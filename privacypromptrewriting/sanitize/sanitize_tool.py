"""
A tool to sanitize sensitive information in given categories.

Flow:
- LLM identifies sensitive info in given categories, produces JSON-structured
    output with sensitive items list
- Agent handler method `sanitize` interprets this JSON msg, gets sensitive items
  as a (value -> category) dict, then sanitizes the sensitive items, and returns
  the sanitized text
"""

from typing import List, Type
from pydantic import BaseModel
from langroid.agent.tool_message import ToolMessage

class SensitiveItem(BaseModel):
    value: str
    category: str

class SanitizeTool(ToolMessage):
    request: str = "sanitize"
    purpose: str = """
            To detect sensitive information in a given body of text, 
            from the given sensitive categories, and send the sensitive items 
            in the <sensitive_items> field in the required format.
            """
    sensitive_items: List[SensitiveItem]


    @classmethod
    def create(cls, sensitive_categories: List[str]) -> Type["SanitizeTool"]:
        class SanitizeToolForCategories(cls):  # type: ignore
            categories: List[str] = sensitive_categories

        return SanitizeToolForCategories

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(
                sensitive_items=[
                    SensitiveItem(value="John Doe", category="Name"),
                    SensitiveItem(value="123-45-6789", category="SSN"),
                    SensitiveItem(value="March 10th", category="Date"),
                ]
            )
        ]

    # @classmethod
    # def instructions(cls) -> str:
    #     return f"""
    #     You will use this tool/function to present detected sensitive information
    #     in the given body of text. You must only focus on these sensitive categories:
    #     {",".join(cls.default_value("categories"))}
    #     You will send the sensitive items by setting the `sensitive_items` field
    #     of this tool/function in the required format.
    #     """

