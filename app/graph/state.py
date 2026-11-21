from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict, total=False):
    repo_url: str
    task: str
    plan: list[str]                                   # planner ke steps
    messages: Annotated[list[BaseMessage], add_messages]  # LLM + tool messages
    files_read: list[str]
    pending_action: dict | None                       # write action waiting for approval
    iterations: int                                   # loop guard
    reflections: int                                  # how many times reflect sent work back
    tool_calls: int                                   # metric for eval
    final_answer: str | None
    stopped_by_guard: bool


class Plan(BaseModel):
    steps: list[str] = Field(description="3-6 concrete steps", min_length=1, max_length=8)


class Reflection(BaseModel):
    complete: bool = Field(description="Answer covers every part of the task and plan")
    grounded: bool = Field(description="Claims are supported by files that were read")
    feedback: str = Field(default="", description="What is missing, if anything")
