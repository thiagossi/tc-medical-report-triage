"""Formatos de entrada e saída da API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """O que o cliente envia."""

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=1000,
        examples=["My chest hurts and I can't breathe"],
    )


class PredictResponse(BaseModel):
    """O que a API devolve."""

    label: Literal["normal", "atencao", "urgente"]
    confidence: float
    probabilities: dict[str, float]
    model_version: str
