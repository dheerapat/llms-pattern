import os
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

class Population(BaseModel):
    """Defines the structure for the study's population."""
    summary: str = Field(description="A concise summary of the study population.")
    inclusion_criteria: List[str] = Field(description="List of key inclusion criteria.")
    exclusion_criteria: List[str] = Field(description="List of key exclusion criteria.")
    sample_size: int = Field(description="The total number of participants in the study.")

class Intervention(BaseModel):
    """Defines the structure for the study's intervention."""
    summary: str = Field(description="A concise summary of the intervention, including drug, dose, and frequency if applicable.")
    details: List[str] = Field(description="A list of specific details describing the intervention.")

class Comparison(BaseModel):
    """Defines the structure for the study's comparison/control group."""
    summary: str = Field(description="A concise summary of the control group, such as placebo, standard of care, or another treatment.")
    details: List[str] = Field(description="A list of specific details describing the comparison.")

class Outcome(BaseModel):
    """Defines the structure for the study's outcomes."""
    primary_outcomes: List[str] = Field(description="List of the primary outcomes or endpoints measured.")
    secondary_outcomes: List[str] = Field(description="List of the secondary outcomes or endpoints measured.")
    results: str = Field(description="A summary of the main results and findings of the study.")

def get_population(text: str) -> Population:
    """Extracts the study population (P) from the text."""
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **POPULATION**.

From the text provided, identify:
1.  **Summary**: A brief description of the participants (e.g., age, condition).
2.  **Inclusion Criteria**: The key criteria participants had to meet.
3.  **Exclusion Criteria**: The key criteria that disqualified participants.
4.  **Sample Size**: The total number of participants enrolled.

Focus exclusively on who was studied. Ignore the intervention, outcomes, and other details.
Present the information in the requested structured format."""
            },
            {"role": "user", "content": f"Here is the clinical study text:\\n\\n{text}"},
        ],
        response_format=Population,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse Population data from the model.")
    return result

def get_intervention(text: str) -> Intervention:
    """Extracts the study intervention (I) from the text."""
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **INTERVENTION**.

From the text provided, identify:
1.  **Summary**: A brief description of the intervention being tested (e.g., drug name, dosage, therapy type).
2.  **Details**: Specific details about how the intervention was administered (e.g., duration, frequency, method).

Focus exclusively on the treatment or test being applied to the intervention group. Ignore the population, comparison group, and outcomes.
Present the information in the requested structured format."""
            },
            {"role": "user", "content": f"Here is the clinical study text:\\n\\n{text}"},
        ],
        response_format=Intervention,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse Intervention data from the model.")
    return result

def get_comparison(text: str) -> Comparison:
    """Extracts the study comparison/control (C) from the text."""
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **COMPARISON** or **CONTROL** group.

From the text provided, identify:
1.  **Summary**: A brief description of what the intervention was compared against (e.g., placebo, standard of care, another drug).
2.  **Details**: Specific details about the comparison treatment.

Focus exclusively on the control group. Ignore the population, intervention group, and outcomes. If there is no control group, state that clearly.
Present the information in the requested structured format."""
            },
            {"role": "user", "content": f"Here is the clinical study text:\\n\\n{text}"},
        ],
        response_format=Comparison,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse Comparison data from the model.")
    return result

def get_outcome(text: str) -> Outcome:
    """Extracts the study outcome (O) from the text."""
    response = client.beta.chat.completions.parse(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **OUTCOMES**.

From the text provided, identify:
1.  **Primary Outcomes**: The main endpoints the study was designed to measure.
2.  **Secondary Outcomes**: Any other endpoints that were measured.
3.  **Results**: A summary of the key findings and conclusions of the study. What happened?

Focus exclusively on what was measured and the results. Ignore the population, intervention, and comparison details.
Present the information in the requested structured format."""
            },
            {"role": "user", "content": f"Here is the clinical study text:\\n\\n{text}"},
        ],
        response_format=Outcome,
        temperature=0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse Outcome data from the model.")
    return result
