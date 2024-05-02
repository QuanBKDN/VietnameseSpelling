from typing import List

from pydantic import BaseModel, Field


class RequestItem(BaseModel):
    content: str = Field(description="A text need to be corrected")


class ResponseItem(BaseModel):
    content: str = Field(description="The original text")
    detection_result: str = Field(description="Errors was detected in original text")
    correction_result: str = Field(description="Text after corrected")
    elapsed_time: float = Field(description="Time to execute")


class BaseResponseItem(BaseModel):
    content: str = Field(description="The original text")
    result: str = Field(description="Text after corrected")
    suggestion: List = Field(description="Suggestion word by word")
    elapsed_time: float = Field(description="Time to execute")


class BatchRequestItem(BaseModel):
    content: List[str] = Field(description="A text need to be corrected")


class BatchResponseItem(BaseModel):
    content: List[str] = Field(description="The original text")
    result: List[str] = Field(description="Text after corrected")
    suggestion: List = Field(description="Suggestion word by word")
    elapsed_time: float = Field(description="Time to execute")


class ExceptionResponseItem(BaseModel):
    message: str = Field(description="Exception message", default="Error")


class LogItem(BaseModel):
    input_text: str
    predict_text: str
