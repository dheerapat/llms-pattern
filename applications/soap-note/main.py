import json
import textwrap
from rich import print
from subjective_data import get_subjective_data, SubjectiveDataResult
from objective_data import get_objective_data, ObjectiveDataResult


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


cc = textwrap.dedent(
    """\
55-year-old male with type 1 diabetes found by family acting confused and 'not himself' for the past day.
Family reports he's been complaining of feeling weak and thirsty,drinking lots of water, and urinating frequently.
He missed his insulin doses yesterday because he was vomiting and 'didn't want to eat.'
He has a 20-year history of diabetes, usually well-controlled. Takes insulin glargine 25 units at bedtime and lispro with meals.
Had gastroenteritis 3 days ago. On presentation, patient is lethargic but arousable, appears dehydrated.
Vital signs: BP 95/60, HR 118, RR 28 (deep breathing pattern), T 37.5°C, O2 sat 98% RA. Mucous membranes dry, skin tenting present.
Breath has fruity odor. Abdominal exam shows mild diffuse tenderness. Neurologically intact but sluggish responses.
Labs: glucose 485 mg/dL, ketones 4.2 mmol/L (normal <0.6), pH 7.12, bicarbonate 8 mEq/L, anion gap 24. Urinalysis shows 4+ glucose, 3+ ketones.
ECG shows sinus tachycardia.""")

print(f"[bold blue]chief complaint:[/bold blue] [white]{cc}[/white]")

sub_resp = get_subjective_data(cc)
obj_resp = get_objective_data(cc)

sub_json_candidates = find_json_objects(sub_resp)
obj_json_candidates = find_json_objects(obj_resp)

sub_json_obj = json.loads(sub_json_candidates[0])
obj_json_obj = json.loads(obj_json_candidates[0])

print("[bold blue]subjective data: [/bold blue]")
print(sub_json_obj)
print("[bold blue]objective data: [/bold blue]")
print(obj_json_obj)
