import os
import json
import textwrap
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

ROUTING_INSTRUCTION = """\
You are a helpful assistant designed to find the best suited route.
You are provided with route description within <routes></routes> XML tags:
<routes>

{routes}

</routes>

<abstract>

{abstract}

</abstract>
"""

GET_ROUTE_PROMPT = """\
You are an expert medical researcher specializing in clinical trial methodology and systematic review analysis.
Your task is to classify a given abstract in <abstract></abstract> XML tags into one of four categories based on the study design and methodology.
You only response the name of the route that best matches the user's request, use the exact name in the <routes></routes>.

Based on your analysis, provide your response in the following JSON formats:
{"route": "route_name"} 
"""

route_config = [
    {
        "name": "randomized_controlled_trial",
        "description": textwrap.dedent(
            """\
        A study must have ALL of the following criteria to be classified as randomized controlled trials:
        - Use a conservative approach with this category. If not sure, don't answer this
        - Random assignment/randomization of participants to intervention groups
        - Human subjects (not animals, cells, or in vitro studies)
        - A control or comparison group (placebo, standard care, or active comparator)
        - An intervention being tested (drug, procedure, therapy, etc.)
        - Prospective design with outcome measurement"""
        ),
    },
    {
        "name": "meta_analysis",
        "description": textwrap.dedent(
            """\
        A study must have ALL of the following criteria to be classified as meta-analysis:
        - Use a conservative approach with this category. If not sure, don't answer this
        - Systematic review methodology with clearly defined search strategy
        - Quantitative synthesis/statistical pooling of results from multiple studies
        - Analysis combining numerical data from at least 2 independent RCT studies
        - Forest plots, pooled effect sizes, or combined statistical measures
        - Assessment of study quality or risk of bias"""
        ),
    },
    {
        "name": "animal_studies",
        "description": textwrap.dedent(
            """\
        A study must have ALL of the following criteria to be classified as animal study:
        - Primary subjects are animals (rodents, primates, fish, birds, etc.)
        - Experimental design with random assignment to control and treatment groups
        - Quantitative outcomes with statistical analysis of results
        - Investigation of biological mechanisms, drug effects, or interventions"""
        ),
    },
    {
        "name": "review_article",
        "description": textwrap.dedent(
            """\
        Any study that does not meet the strict criteria for the above three categories, including:
        - Narrative reviews or expert opinions without systematic methodology
        - Observational studies (cohort, case-control, cross-sectional)
        - Case reports or case series
        - Editorial or commentary articles
        - Studies that mention other research but don't conduct systematic review or meta-analysis
        - In vitro or cell culture studies
        - Surveys or questionnaire-based studies without experimental intervention"""
        ),
    },
]


class ClassificationResult(BaseModel):
    route: Literal[
        "randomized_controlled_trial",
        "meta_analysis",
        "animal_studies",
        "review_article",
    ]


def format_prompt(route_config: List[Dict[str, str]], abstract: str):
    return (
        ROUTING_INSTRUCTION.format(routes=json.dumps(route_config), abstract=abstract)
        + GET_ROUTE_PROMPT
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
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": route_prompt},
        ],
        # reasoning_effort="low",
        temperature=0.1,
    )

    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)

    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = json.loads(json_candidates[0])
        return ClassificationResult.model_validate(json_obj)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


GET_KEYWORD_PROMPT = """
You are a **biomedical search specialist** focused on generating **optimal PubMed keywords** in JSON format.

## Primary Function

Convert user queries — including **complex clinical vignettes** — into a list of concise, PubMed-optimized keywords for use with the `[Title/Abstract:~5]` proximity operator.

## Keyword Requirements

* **List output**: Return a JSON object with `"keywords"` as a list.
* **2-3 keywords**
* **Distinct facets**: Each keyword should capture a unique aspect (disease, treatment, modifier, comorbidity, context).
* **Medical terminology**: Prefer MeSH and standard biomedical language.
* **PubMed optimized**: Use phrases likely to appear in titles/abstracts.

---

## Optimization Strategy

### Query Analysis

1. **Identify the primary disease/condition**.
2. **Extract relevant modifiers** (comorbidities, allergies, severity, demographics if central).
3. **Determine query focus** (treatment, diagnosis, prognosis, risk, prevention).
4. **Discard irrelevant detail** (lab values, vitals, or unrelated context).
5. **Generate multiple perspectives** (condition, context, intervention, risk).

### Keyword Selection Rules

* **Short**.
* **No stop words** (avoid "and", "or", "of").
* **Established terms** (use “myocardial infarction” not “heart attack”).
* **Prefer common PubMed usage** (e.g., “COPD”, “HIV”, “AIDS”).
* **Distinct entries**: Do not combine unrelated concepts into a single keyword.

---

## Few-Shot Examples

**Example 1**
User Query:
*"Side effects of statins on muscles"*

Output:

```json
{
  "keywords": [
    "statin myopathy",
    "statin adverse effects"
  ]
}
```

---

**Example 2**
User Query:
*"How effective is cognitive behavioral therapy for depression?"*

Output:

```json
{
  "keywords": [
    "CBT depression",
    "cognitive therapy"
  ]
}
```

---

**Example 3**
User Query:
*"Cancer treatment using immunotherapy"*

Output:

```json
{
  "keywords": [
    "cancer immunotherapy",
    "oncology immunotherapy"
  ]
}
```

---

**Example 4**
User Query:
*"A patient with tuberculosis develops liver toxicity while on isoniazid. What are the management options?"*

Output:

```json
{
  "keywords": [
    "tuberculosis treatment",
    "isoniazid hepatotoxicity"
  ]
}
```

---

**Example 5**
User Query:
*"An elderly smoker with COPD develops frequent exacerbations despite inhaler therapy. What additional treatments are available?"*

Output:

```json
{
  "keywords": [
    "COPD exacerbation",
    "elderly smoker",
    "COPD treatment"
  ]
}
```

---

## Output Format

Return only JSON in this format:

```json
{
  "keywords": [
    "keyword1",
    "keyword2",
    ...,
  ]
}
```

---

## Quality Checks

* Would each keyword likely appear in PubMed titles/abstracts?
* Does each keyword capture a distinct aspect of the query?
* Are terms concise, precise, and biomedical in nature?
* Do they benefit from the `[Title/Abstract:~N]` proximity operator?

Based on your analysis, provide your response in the following JSON formats:
```json
{
  "keywords": [
    "keyword1",
    "keyword2",
    ...,
  ]
}
```
"""


class KeywordResult(BaseModel):
    keywords: list[str]


def get_keyword(user_query: str):
    completion = client.chat.completions.create(
        model=os.getenv("TEXT_MODEL_NAME", ""),
        messages=[
            {"role": "system", "content": GET_KEYWORD_PROMPT},
            {"role": "user", "content": user_query},
        ],
        # reasoning_effort="low",
        temperature=0.1,
    )
    if completion.choices[0].message.content is None:
        raise ValueError("No response content received from the model")

    json_candidates = find_json_objects(completion.choices[0].message.content)
    if len(json_candidates) == 0:
        raise ValueError("No JSON object found in response")

    try:
        json_obj = KeywordResult.model_validate(json.loads(json_candidates[0]))
        return json_obj
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")


if __name__ == "__main__":
    # test classification
    # ids = []
    # ids += ["36578889", "30039871", "35082662", "35537861", "33999947"]  # meta
    # ids += ["28066101", "34476568", "35939311", "37960261", "35956364"]  # rct
    # ids += ["32457512", "38720498", "21831011", "34706925", "37571305"]  # non rct

    # for id in ids:
    #     print(id)
    #     abs = get_abstract(id, "xml")
    #     parsed = parse_pubmed_xml(abs)
    #     try:
    #         result = classify(parsed[0].abstract.full_abstract)
    #         print(result)
    #     except Exception as e:
    #         print("exception occur")
    #         print(e.args)

    # test keyword extraction
    print(get_keyword("Do CoQ10 supplements provide any cardiovascular benefits?"))
