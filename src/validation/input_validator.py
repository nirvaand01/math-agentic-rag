from pydantic import BaseModel, validator
from typing import Optional

class MathQuery(BaseModel):
    """Class to validate and structure math problem inputs."""
    
    question: str
    subject: str
    type: str
    
    @validator("subject")
    def validate_subject(cls, v):
        """Validate that subject is one of the allowed values."""
        allowed_subjects = ["math", "phy", "chem"]
        if v not in allowed_subjects:
            raise ValueError(f"Subject must be one of {allowed_subjects}")
        return v
    
    @validator("type")
    def validate_type(cls, v):
        """Validate that type is one of the allowed values."""
        allowed_types = ["MCQ", "MCQ(multiple)", "Integer", "Numeric"]
        if v not in allowed_types:
            raise ValueError(f"Type must be one of {allowed_types}")
        return v
    
    @validator("question")
    def validate_question(cls, v):
        """Validate that question is not empty."""
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v 