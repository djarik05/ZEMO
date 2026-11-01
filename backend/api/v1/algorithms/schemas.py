from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class HyperparameterSpecResponse(BaseModel):
    name: str
    param_type: str
    default: Any
    description: str
    options: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allow_none: bool = False


class AlgorithmResponse(BaseModel):
    id: str
    display_name: str
    task: Literal["classification", "regression", "clustering"]
    is_supervised: bool
    description: str
    family: str
    tags: List[str] = Field(default_factory=list)
    hyperparameters: Dict[str, HyperparameterSpecResponse]
    default_hyperparameters: Dict[str, Any]


class AlgorithmListResponse(BaseModel):
    total: int
    task_filter: Optional[str] = None
    algorithms: List[AlgorithmResponse]


class ValidateHyperparametersRequest(BaseModel):
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    task: Optional[str] = None


class ValidateHyperparametersResponse(BaseModel):
    valid: bool
    algorithm_id: str
    display_name: str
    sanitized_hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


class AlgorithmCompatibilityRequest(BaseModel):
    task: str
    target_type: Optional[str] = None


class AlgorithmCompatibilityResponse(BaseModel):
    compatible: bool
    algorithm_id: str
    display_name: str
    task: str
    errors: List[str] = Field(default_factory=list)
