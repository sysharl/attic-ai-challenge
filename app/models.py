from pydantic import BaseModel


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]  # [{"page": int, "text": str}]


class UploadResponse(BaseModel):
    session_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
