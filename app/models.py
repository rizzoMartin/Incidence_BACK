from pydantic import BaseModel, Field
from enum import Enum

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class AnalyzeIn(BaseModel):
    solicitud: str = Field(..., example="Me gusta este producto")

class AnalyzeOut(BaseModel):
    suggested_response: str = Field(..., example="Estamos trabajando en su solicitud y le responderemos cuanto antes.")
    sentiment: Sentiment = Field(..., example="positive")
    urgency: Urgency = Field(..., example="low")
    tags: list[str] = Field(..., example=["urgente", "error"])

class ChatMessage(BaseModel):
    role: str  # 'user' o 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]  # últimos 5 mensajes
    user_message: str

class ChatResponse(BaseModel):
    response: str