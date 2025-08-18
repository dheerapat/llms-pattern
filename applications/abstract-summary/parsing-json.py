import os
import json
from openai import OpenAI
from typing import Dict, List, Literal
from dotenv import load_dotenv
from rich import print
from pubmed import get_abstract, parse_pubmed_xml
from pydantic import BaseModel, ValidationError

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

TASK_INSTRUCTION = """
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<abstract>

{abstract}

</abstract>
"""

FORMAT_PROMPT = """
You are an expert medical researcher specializing in clinical trial methodology and systematic review analysis.
Your task is to classify a given abstract in <abstract></abstract> XML tags into one of four categories based on the study design and methodology.
You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats:
{"route": "route_name"} 
"""

route_config = [
    {
        "name": "rct",
        "description": """
        A study must have ALL of the following criteria to be classified as RCT:
        - Use a conservative approach with this category. If not sure, don't answer this
        - Random assignment/randomization of participants to intervention groups
        - Human subjects (not animals, cells, or in vitro studies)
        - A control or comparison group (placebo, standard care, or active comparator)
        - An intervention being tested (drug, procedure, therapy, etc.)
        - Prospective design with outcome measurement""".strip(),
    },
    {
        "name": "meta_analysis",
        "description": """
        A study must have ALL of the following criteria to be classified as meta-analysis:
        - Use a conservative approach with this category. If not sure, don't answer this
        - Systematic review methodology with clearly defined search strategy
        - Quantitative synthesis/statistical pooling of results from multiple studies
        - Analysis combining numerical data from at least 2 independent RCT studies
        - Forest plots, pooled effect sizes, or combined statistical measures
        - Assessment of study quality or risk of bias""".strip(),
    },
    {
        "name": "animal_studies",
        "description": """
        A study must have ALL of the following criteria to be classified as animal study:
        - Primary subjects are animals (rodents, primates, fish, birds, etc.)
        - Experimental design with random assignment to control and treatment groups
        - Quantitative outcomes with statistical analysis of results
        - Investigation of biological mechanisms, drug effects, or interventions""".strip(),
    },
    {
        "name": "review_article",
        "description": """
        Any study that does not meet the strict criteria for the above three categories, including:
        - Narrative reviews or expert opinions without systematic methodology
        - Observational studies (cohort, case-control, cross-sectional)
        - Case reports or case series
        - Editorial or commentary articles
        - Studies that mention other research but don't conduct systematic review or meta-analysis
        - In vitro or cell culture studies
        - Surveys or questionnaire-based studies without experimental intervention""".strip(),
    },
]


class ClassificationResult(BaseModel):
    route: Literal["rct", "meta_analysis", "animal_studies", "review_article"]


def format_prompt(route_config: List[Dict[str, str]], abstract: str):
    return (
        TASK_INSTRUCTION.format(routes=json.dumps(route_config), abstract=abstract)
        + FORMAT_PROMPT
    )


def find_json_objects(text: str) -> List[str]:
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


def classify(abstract_text: str) -> ClassificationResult:
    route_prompt = format_prompt(route_config, abstract_text)
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            # {"role": "system", "content": "/no_think"},
            {"role": "user", "content": route_prompt},
        ],
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)
    if not json_candidates:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return ClassificationResult.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


if __name__ == "__main__":
    ids = []
    ids += ["36578889", "30039871", "35082662", "35537861", "33999947"]  # meta
    ids += ["28066101", "34476568", "35939311", "37960261", "35956364"]  # rct
    ids += ["32457512", "38720498", "21831011", "34706925", "37571305"]  # non rct

    for id in ids:
        print(id)
        abs = get_abstract(id, "xml")
        parsed = parse_pubmed_xml(abs)
        result = classify(parsed[0].abstract.full_abstract)
        print(result)
