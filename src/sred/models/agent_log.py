from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel
from sred.models.base import TimestampMixin


class ToolCallLog(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    session_id: Optional[str] = Field(default=None, index=True, description="UUID grouping calls from one agent loop invocation")

    tool_name: str
    arguments_json: str  # JSON string of tool arguments
    result_json: str  # JSON string of tool result
    success: bool
    duration_ms: int  # Execution time in milliseconds


class LLMCallLog(TimestampMixin, table=True):
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    session_id: Optional[str] = Field(default=None, index=True, description="UUID grouping calls from one agent loop invocation")

    model: str
    prompt_summary: str  # Truncated/summarised prompt for audit
    message_count: int  # Number of messages in the conversation
    tool_calls_count: int = Field(default=0)

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)

    finish_reason: Optional[str] = None
