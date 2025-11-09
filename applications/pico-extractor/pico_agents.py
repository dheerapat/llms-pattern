import os
import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

_ = load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

POPULATION_SYSTEM_PROMPT = """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **POPULATION**.

From the text provided, identify:
1.  **Summary**: A brief description of the participants (e.g., age, condition).
2.  **Inclusion Criteria**: The key criteria participants had to meet.
3.  **Exclusion Criteria**: The key criteria that disqualified participants.
4.  **Sample Size**: The total number of participants enrolled.

Focus exclusively on who was studied. Ignore the intervention, outcomes, and other details.
Present the information in the requested structured format.

Example response:
{
  "summary": "Adults aged 18-75 with type 2 diabetes mellitus",
  "inclusion_criteria": ["Age 18-75 years", "Diagnosed with type 2 diabetes", "HbA1c between 7.0-10.0%"],
  "exclusion_criteria": ["Pregnant or lactating women", "Severe renal impairment", "Current insulin therapy"],
  "sample_size": 250
}"""

INTERVENTION_SYSTEM_PROMPT = """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **INTERVENTION**.

From the text provided, identify:
1.  **Summary**: A brief description of the intervention being tested (e.g., drug name, dosage, therapy type).
2.  **Details**: Specific details about how the intervention was administered (e.g., duration, frequency, method).

Focus exclusively on the treatment or test being applied to the intervention group. Ignore the population, comparison group, and outcomes.
Present the information in the requested structured format.

Example response:
{
  "summary": "Metformin 500mg twice daily oral medication",
  "details": ["500mg tablets taken orally", "Twice daily with meals", "Treatment duration: 24 weeks", "Dose titration allowed"]
}"""

COMPARISON_SYSTEM_PROMPT = """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **COMPARISON** or **CONTROL** group.

From the text provided, identify:
1.  **Summary**: A brief description of what the intervention was compared against (e.g., placebo, standard of care, another drug).
2.  **Details**: Specific details about the comparison treatment.

Focus exclusively on the control group. Ignore the population, intervention group, and outcomes. If there is no control group, state that clearly.
Present the information in the requested structured format.

Example response:
{
  "summary": "Placebo tablets matching intervention in appearance",
  "details": ["Identical-looking placebo tablets", "Same dosing schedule as intervention", "No active ingredients", "Administered orally"]
}"""

OUTCOME_SYSTEM_PROMPT = """You are an expert medical researcher specializing in analyzing clinical trial documents.
Your task is to extract information about the study's **OUTCOMES**.

From the text provided, identify:

1.  **Primary Outcomes**: The main endpoints the study was designed to measure.
    - These are typically explicitly labeled as "primary outcome(s)", "primary endpoint(s)", or "primary objective(s)"
    - Look for phrases like: "The primary outcome was...", "Primary endpoints included...", "The main outcome measure..."
    - Primary outcomes are usually fewer in number (often 1-3) and represent the most important questions the study aims to answer
    - If multiple outcomes are listed together without clear hierarchy, the first one mentioned is often primary

2.  **Secondary Outcomes**: Any other endpoints that were measured.
    - These are typically labeled as "secondary outcome(s)", "secondary endpoint(s)", or "exploratory outcomes"
    - Look for phrases like: "Secondary outcomes included...", "Other endpoints measured...", "Additional analyses examined..."
    - Secondary outcomes support or provide additional context to the primary findings
    - May include subgroup analyses, safety measures, or quality of life assessments

3.  **Results**: A summary of the key findings and conclusions of the study.
    - What were the measured values for primary outcomes? Include effect sizes, percentages, or mean changes
    - Were results statistically significant? Include p-values and confidence intervals when available
    - How did treatment groups compare?
    - What were the main conclusions drawn by the study authors?

**Important distinctions:**
- DO NOT confuse the study population/inclusion criteria with outcomes
- DO NOT confuse the intervention or treatment being tested with what was measured
- Outcomes are what was MEASURED or ASSESSED, not what was DONE to participants

**If the document does not clearly distinguish between primary and secondary outcomes:**
- Place the most prominently featured or first-mentioned endpoint(s) as primary
- Place supporting or additional measures as secondary
- When truly uncertain, note this in your response

Focus exclusively on what was measured and the results. Ignore the population, intervention, and comparison details.
Present the information in the requested structured format.

Example response:
{
  "primary_outcomes": ["Change in HbA1c levels from baseline at 6 months", "Incidence of hypoglycemic events"],
  "secondary_outcomes": ["Weight change from baseline", "Quality of life scores (SF-36)", "Lipid profile improvements", "Treatment adherence rates"],
  "results": "Metformin group showed significant reduction in HbA1c (-1.2%) compared to placebo (-0.1%, p<0.001). No severe hypoglycemic events reported in either group. Secondary analyses showed modest weight loss in metformin group (-2.3 kg vs +0.5 kg, p=0.02) and improved LDL cholesterol levels."
}"""


class Population(BaseModel):
    summary: str
    inclusion_criteria: list[str]
    exclusion_criteria: list[str]
    sample_size: int


class Intervention(BaseModel):
    summary: str
    details: list[str]


class Comparison(BaseModel):
    summary: str
    details: list[str]


class Outcome(BaseModel):
    primary_outcomes: list[str]
    secondary_outcomes: list[str]
    results: str


def find_json_objects(text: str) -> list[str]:
    json_candidates = []
    i = 0
    while i < len(text):
        if text[i] == "{":
            brace_count = 1
            start = i
            i += 1

            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                i += 1

            if brace_count == 0:
                json_candidates.append(text[start:i])
        else:
            i += 1

    return json_candidates


def get_population(text: str) -> Population:
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": POPULATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here is the clinical study text:\\n\\n{text}",
            },
        ],
        temperature=0,
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)

    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return Population.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


def get_intervention(text: str) -> Intervention:
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": INTERVENTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here is the clinical study text:\\n\\n{text}",
            },
        ],
        temperature=0,
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)

    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return Intervention.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


def get_comparison(text: str) -> Comparison:
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": COMPARISON_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here is the clinical study text:\\n\\n{text}",
            },
        ],
        temperature=0,
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)

    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return Comparison.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


def get_outcome(text: str) -> Outcome:
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {
                "role": "system",
                "content": OUTCOME_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Here is the clinical study text:\\n\\n{text}",
            },
        ],
        temperature=0,
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)

    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return Outcome.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")
