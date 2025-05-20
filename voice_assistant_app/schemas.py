from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator
from datetime import date

class UserFormData(BaseModel):
    """Pydantic model defining the form fields and validation rules."""
    
    full_name: str = Field(
        ..., 
        description="User's full name",
        min_length=2,
        max_length=100
    )
    
    email: str = Field(
        ..., 
        description="User's email address",
        pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )
    
    age: int = Field(
        ..., 
        description="User's age in years",
        ge=18,
        le=120
    )
    
    occupation: str = Field(
        ..., 
        description="User's current job or profession",
        min_length=2,
        max_length=100
    )
    
    experience_level: Literal["Beginner", "Intermediate", "Advanced", "Expert"] = Field(
        ...,
        description="User's experience level in their field"
    )
    
    preferred_language: Literal["Python", "JavaScript", "Java", "C++", "Go", "Rust", "Other"] = Field(
        ...,
        description="User's preferred programming language"
    )
    
    project_interests: List[str] = Field(
        ...,
        description="List of project interests or goals",
        min_items=1,
        max_items=5
    )
    
    availability_per_week: int = Field(
        ...,
        description="Hours available per week for the project",
        ge=1,
        le=168
    )
    
    start_date: date = Field(
        ...,
        description="Preferred project start date"
    )
    
    additional_notes: Optional[str] = Field(
        None,
        description="Any additional information or special requirements",
        max_length=500
    )
    
    @validator('project_interests')
    def validate_project_interests(cls, v):
        """Ensure each project interest has a reasonable length."""
        for interest in v:
            if len(interest) < 2 or len(interest) > 100:
                raise ValueError("Each project interest must be between 2 and 100 characters")
        return v
    
    class Config:
        """Configuration for the Pydantic model."""
        json_schema_extra = {
            "example": {
                "full_name": "Jane Doe",
                "email": "jane.doe@example.com",
                "age": 30,
                "occupation": "Software Developer",
                "experience_level": "Intermediate",
                "preferred_language": "Python",
                "project_interests": ["Web Development", "Machine Learning", "Data Visualization"],
                "availability_per_week": 20,
                "start_date": "2025-06-01",
                "additional_notes": "Looking for collaborative projects with flexible hours."
            }
        }


class FormState(BaseModel):
    """Model to track the state of form completion."""
    
    current_field: str = Field(
        default="full_name",
        description="The field currently being populated"
    )
    
    completed_fields: List[str] = Field(
        default_factory=list,
        description="List of fields that have been completed"
    )
    
    field_values: dict = Field(
        default_factory=dict,
        description="Dictionary of field names and their values"
    )
    
    confirmation_state: bool = Field(
        default=False,
        description="Whether we are in a confirmation state for the current field"
    )
    
    extraction_attempts: int = Field(
        default=0,
        description="Number of attempts to extract a valid value for the current field"
    )
    
    is_complete: bool = Field(
        default=False,
        description="Whether all required fields are complete"
    )