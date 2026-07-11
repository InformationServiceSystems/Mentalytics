import altair as alt
import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from html import escape
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# --------------------------------------------------
# Constants
# --------------------------------------------------
DATASET_PATH = Path("../Datasets/merged_dataset_with_ids_and_extra_exercises.json")

GENERAL_NORMS = {
    "Agreeableness": 5.23,
    "Conscientiousness": 5.4,
    "Extraversion": 4.44,
    "Emotional Stability": 4.83,
    "Openness": 5.38,
}

BIG5_TO_ICF_BPS_DOMAIN = {
    "Agreeableness": "Social cooperation / support",
    "Conscientiousness": "Activity self-management",
    "Extraversion": "Participation engagement",
    "Emotional Stability": "Emotional regulation",
    "Openness": "Adaptability to rehabilitation",
}

MODEL_ID = "../models/Llama-3.1-8B-Instruct"

USE_LLAMA_FOR_RADAR = True
USE_LLAMA_FOR_MATRIX = True

BIG5_TRAITS = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Emotional Stability",
    "Openness",
]


# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Mentalytics Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "selected_patient_id" not in st.session_state:
    st.session_state.selected_patient_id = None

if "user_data" not in st.session_state:
    st.session_state.user_data = None

if "llama_loaded" not in st.session_state:
    st.session_state.llama_loaded = False

# --------------------------------------------------
# Data loading
# --------------------------------------------------
def load_patient_dataset():
    """
    Loads the full JSON dataset once and indexes it by metadata.ui_id.

    Expected JSON structure:
    [
        {
            "metadata": {
                "ui_id": "AMM_0001"
            },
            ...
        },
        ...
    ]
    """

    if not DATASET_PATH.exists():
        st.error(f"Dataset file not found: {DATASET_PATH}")
        return {}

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records_by_id = {}

    for record in data:
        metadata = record.get("metadata", {})
        ui_id = metadata.get("ui_id")

        if ui_id:
            records_by_id[ui_id] = record

    return records_by_id


def load_patient_by_id(patient_id: str):
    """
    Loads one patient record using the typed AMM ID.
    Returns the raw nested JSON record.
    """

    if not patient_id:
        return None

    patient_id = patient_id.strip()

    records_by_id = load_patient_dataset()

    if patient_id not in records_by_id:
        return None

    return records_by_id[patient_id]



# --------------------------------------------------
# Global styling
# --------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp {
            background-color: #F6F8FB;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }

        section[data-testid="stSidebar"] .stRadio > label {
            display: none;
        }

        /* Page title */
        .main-title {
            font-size: 34px;
            font-weight: 800;
            color: #0B2E59;
            margin-bottom: 4px;
        }

        .subtitle {
            font-size: 16px;
            color: #64748B;
            margin-bottom: 25px;
        }

        /* Cards */
        .card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }

        .card-title {
            font-size: 20px;
            font-weight: 750;
            color: #0B2E59;
            margin-bottom: 15px;
        }

        .metric-label {
            font-size: 13px;
            color: #64748B;
            margin-bottom: 4px;
        }

        .metric-value {
            font-size: 23px;
            font-weight: 750;
            color: #0F172A;
        }

        .small-muted {
            font-size: 13px;
            color: #64748B;
        }

        .placeholder-box {
            height: 280px;
            border: 2px dashed #CBD5E1;
            border-radius: 16px;
            background-color: #F8FAFC;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748B;
            font-size: 17px;
            font-weight: 600;
            text-align: center;
        }

        .placeholder-box-small {
            height: 180px;
            border: 2px dashed #CBD5E1;
            border-radius: 16px;
            background-color: #F8FAFC;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748B;
            font-size: 16px;
            font-weight: 600;
            text-align: center;
        }

        .status-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background-color: #E0F2FE;
            color: #075985;
            font-weight: 700;
            font-size: 14px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .warning-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background-color: #FEF3C7;
            color: #92400E;
            font-weight: 700;
            font-size: 14px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        .danger-pill {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background-color: #FEE2E2;
            color: #991B1B;
            font-weight: 700;
            font-size: 14px;
            margin-right: 8px;
            margin-bottom: 8px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            padding: 15px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            border-radius: 18px;
        }

        /* Hide Streamlit default menu/footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )



# --------------------------------------------------
# Sidebar
# --------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 4px 24px 4px;">
                <div style="
                    width: 54px;
                    height: 54px;
                    border-radius: 16px;
                    background: linear-gradient(135deg, #0B5CAD, #38BDF8);
                    color: white;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 28px;
                    font-weight: 800;
                    margin-bottom: 10px;
                ">
                    +
                </div>
                <div style="font-size: 22px; font-weight: 800; color: #0B2E59;">
                    Mentalytics
                </div>
                <div style="font-size: 13px; color: #64748B;">
                    Clinical AMM Dashboard
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        page = st.radio(
            "",
            [
                "Dashboard",
                "Patient Profile",
                "Physical Insights",
                "Psychological Insights"
                # "Reports"
            ],
            index=0
        )

        st.markdown("---")

        if st.session_state.selected_patient_id:
            st.markdown(
                f"""
                <div style="font-size: 13px; color: #64748B;">Loaded patient</div>
                <div style="font-size: 18px; font-weight: 800; color: #0B2E59;">
                    {st.session_state.selected_patient_id}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="font-size: 13px; color: #64748B;">
                    No patient loaded
                </div>
                """,
                unsafe_allow_html=True
            )

    return page



# --------------------------------------------------
# General helpers
# --------------------------------------------------
def translate_yes_no(value):
    text = str(value).strip().lower()

    if text in ["ja", "yes"]:
        return "Yes"

    if text in ["nein", "no"]:
        return "No"

    return str(value)


def translate_german_value(value):
    """
    Small display translator for common German survey values.
    Keeps the original score number where useful.
    """

    if value is None:
        return "-"

    text = str(value).strip()

    replacements = {
        "Ja": "Yes",
        "Nein": "No",
        "Männlich": "Male",
        "Weiblich": "Female",
        "Divers": "Other",
        "Student": "Student",
        "Entspannt": "Relaxed",
        "Gestresst": "Stressed",
        "Glücklich": "Happy",
        "Traurig": "Sad",
        "Sehr gut": "Very good",
        "Gut": "Good",
        "Befriedigend": "Satisfactory",
        "Ausreichend": "Adequate",
        "Schlecht": "Poor",
        "Vollständig": "Complete",
        "Durchnittlich schwierig": "Moderately difficult",
        "Durchschnittlich schwierig": "Moderately difficult",
        "Sehr schwierig": "Very difficult",
        "Leicht schwierig": "Slightly difficult",
        "Nicht schwierig": "Not difficult",
        "Extrem schwierig": "Extremely difficult",
        "Tage": "days",
        "Tag": "day",
        "Wochen": "weeks",
        "Woche": "week",
        "Mehr als einen Monat": "More than one month",
        "30-60 Minuten": "30-60 minutes",
        "60-90 Minuten": "60-90 minutes",
        "Mehr als 90 Minuten": "More than 90 minutes",
    }

    translated = text

    for german, english in replacements.items():
        translated = translated.replace(german, english)

    return translated


def extract_numeric_score(value):
    """
    Converts values like:
    '3 - Durchnittlich schwierig'
    '4 - Sehr schwierig'
    3
    into integer scores.
    """
    if value is None:
        return 0

    text = str(value).strip()

    if text == "-" or text == "":
        return 0

    match = re.search(r"\d+", text)

    if match:
        return int(match.group())

    return 0


def yes_no_value(value):
    text = str(value).strip().lower()

    if "ja" in text or "yes" in text:
        return True

    if "nein" in text or "no" in text:
        return False

    return False


def clamp_score(value, min_value=1, max_value=5):
    return max(min_value, min(max_value, int(round(value))))



# --------------------------------------------------
# Patient field accessors
# --------------------------------------------------
def get_patient_id(user_data):
    return user_data.get("metadata", {}).get("ui_id", "-")


def get_age(user_data):
    return user_data.get("demographics", {}).get("age", "-")


def get_gender(user_data):
    return user_data.get("demographics", {}).get("gender", "-")


def get_employment(user_data):
    return user_data.get("demographics", {}).get("employment", "-")


def get_overall_health(user_data):
    return user_data.get("health", {}).get("overall_health", "-")


def get_mobility(user_data):
    return user_data.get("health", {}).get("mobility", "-")


def get_disability(user_data):
    return user_data.get("health", {}).get("disability", "-")


def get_activities(user_data):
    return user_data.get("lifestyle", {}).get("activities", "-")


def get_exercise_days(user_data):
    return user_data.get("lifestyle", {}).get("days_per_week", "-")


def get_emotional_state(user_data):
    return user_data.get("psychology", {}).get("emotional", "-")


def get_main_perceived_difficulty(user_data):
    return extract_numeric_score(
        user_data.get("exercise_task", {}).get("perceived_difficulty", "-")
    )


def get_main_actual_difficulty(user_data):
    return extract_numeric_score(
        user_data.get("exercise_task", {}).get("actual_difficulty", "-")
    )



# --------------------------------------------------
# Big Five and ICF/BPS helpers
# --------------------------------------------------
def reverse_score(value):
    """
    Reverse-codes Big Five values on a 1-7 scale.
    Example: 1 -> 7, 2 -> 6, ..., 7 -> 1
    """
    score = extract_numeric_score(value)

    if score == 0:
        return 0

    return 8 - score


def average_valid(scores):
    valid_scores = [score for score in scores if score > 0]

    if not valid_scores:
        return 0

    return sum(valid_scores) / len(valid_scores)


def get_big5_scores(user_data):
    """
    Calculates Big Five personality scores from the raw nested JSON record.

    Each trait is based on two survey items:
    one direct item and one reverse-coded opposite item.
    """

    psychology = user_data.get("psychology", {})
    big5 = psychology.get("big5", {})

    agreeableness = average_valid([
        reverse_score(big5.get("quarrel", "-")),
        extract_numeric_score(big5.get("warm", "-"))
    ])

    conscientiousness = average_valid([
        extract_numeric_score(big5.get("discipline", "-")),
        reverse_score(big5.get("careless", "-"))
    ])

    extraversion = average_valid([
        extract_numeric_score(big5.get("extrav", "-")),
        reverse_score(big5.get("quiet", "-"))
    ])

    emotional_stability = average_valid([
        reverse_score(big5.get("anxious", "-")),
        extract_numeric_score(big5.get("stable", "-"))
    ])

    openness = average_valid([
        extract_numeric_score(big5.get("open", "-")),
        reverse_score(big5.get("uncreative", "-"))
    ])

    return {
        "Agreeableness": round(agreeableness, 2),
        "Conscientiousness": round(conscientiousness, 2),
        "Extraversion": round(extraversion, 2),
        "Emotional Stability": round(emotional_stability, 2),
        "Openness": round(openness, 2)
    }


def get_icf_bps_domain_profile_df(user_data, big5_scores=None):
    """
    Creates a grouped dataframe comparing the patient's psychological domain profile
    with general Big Five norm values.
    """

    if big5_scores is None:
        big5_scores = get_big5_scores(user_data)

    rows = []

    for trait, domain in BIG5_TO_ICF_BPS_DOMAIN.items():
        rows.append({
            "Domain": domain,
            "Trait": trait,
            "Legend": "Patient",
            "Score": big5_scores.get(trait, 0)
        })

        rows.append({
            "Domain": domain,
            "Trait": trait,
            "Legend": "General Norm",
            "Score": GENERAL_NORMS.get(trait, 0)
        })

    return pd.DataFrame(rows)

# --------------------------------------------------
# Llama-based Big Five scoring
# --------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@st.cache_resource
def load_llama_model():
    device = get_device()

    if device == "cuda":
        torch_dtype = torch.float16
        device_map = "auto"
    elif device == "mps":
        torch_dtype = torch.float16
        device_map = None
    else:
        torch_dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=True
    )

    if device_map is None:
        model.to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()

    return tokenizer, model, device

def initialize_llama_on_startup():
    """
    Loads Llama as soon as the app opens.
    This avoids waiting for the model only after a patient ID is searched.
    """

    if not USE_LLAMA_FOR_RADAR and not USE_LLAMA_FOR_MATRIX:
        return

    with st.spinner("Loading Llama 3.1 8B model. This may take a moment..."):
        load_llama_model()

    st.session_state.llama_loaded = True

def build_big5_llama_prompt(user_data):
    psychology = user_data.get("psychology", {})
    health = user_data.get("health", {})
    lifestyle = user_data.get("lifestyle", {})

    big5 = psychology.get("big5", {})

    patient_summary = {
        "big5_survey_items": big5,
        "mood_today": psychology.get("mood_today", "-"),
        "stress_or_anxiety_24h": psychology.get("stress_or_anxiety_24h", "-"),
        "emotional": psychology.get("emotional", "-"),
        "overall_health": health.get("overall_health", "-"),
        "mobility": health.get("mobility", "-"),
        "surgery": health.get("surgery", "-"),
        "pt_adherence": health.get("pt_adherence", "-"),
        "activities": lifestyle.get("activities", "-"),
        "days_per_week": lifestyle.get("days_per_week", "-"),
        "session_length": lifestyle.get("session_length", "-"),
    }

    return f"""
You are scoring a rehabilitation dashboard personality profile.

Task:
Given the structured patient survey data, assign scores for the five Big Five traits.

Traits:
- Agreeableness
- Conscientiousness
- Extraversion
- Emotional Stability
- Openness

Rules:
- Return ONLY valid JSON.
- Do not include explanations.
- Each score must be a number from 1.0 to 7.0.
- Use decimal values such as 3.5 if appropriate.
- Do not diagnose the patient.
- Do not generate medical advice.
- Base the scores only on the provided structured survey values.

Patient data:
{json.dumps(patient_summary, ensure_ascii=False, indent=2)}

Required JSON format:
{{
  "Agreeableness": 0,
  "Conscientiousness": 0,
  "Extraversion": 0,
  "Emotional Stability": 0,
  "Openness": 0
}}
""".strip()


def extract_json_object(text):
    if not text:
        return None

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def clamp_big5_score(value):
    try:
        score = float(value)
    except Exception:
        return None

    if score < 1:
        score = 1.0

    if score > 7:
        score = 7.0

    return round(score, 2)


def validate_llama_big5_scores(scores):
    if not isinstance(scores, dict):
        return None

    cleaned = {}

    for trait in BIG5_TRAITS:
        score = clamp_big5_score(scores.get(trait))

        if score is None:
            return None

        cleaned[trait] = score

    return cleaned

def clamp_matrix_score(value):
    try:
        score = int(round(float(value)))
    except Exception:
        return None

    if score < 1:
        score = 1

    if score > 5:
        score = 5

    return score


def validate_llama_matrix_scores(scores):
    if not isinstance(scores, dict):
        return None

    required_exercises = [
        "Squats",
        "Calf Raises",
        "Standing Toe Touch",
        "Toe Touch Stretch",
    ]

    rows = []

    for exercise in required_exercises:
        exercise_scores = scores.get(exercise)

        if not isinstance(exercise_scores, dict):
            return None

        row = {
            "Exercise": exercise
        }

        for col in MATRIX_COLUMNS:
            score = clamp_matrix_score(exercise_scores.get(col))

            if score is None:
                return None

            row[col] = score

        rows.append(row)

    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_llama_big5_scores_cached(patient_id, user_data_json):
    tokenizer, model, device = load_llama_model()

    user_data = json.loads(user_data_json)

    prompt = build_big5_llama_prompt(user_data)

    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only scoring assistant for a rehabilitation dashboard."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=120,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    parsed = extract_json_object(response)
    validated = validate_llama_big5_scores(parsed)

    if validated is None:
        raise ValueError(f"Llama returned invalid Big Five JSON: {response}")

    return validated

@st.cache_data(show_spinner=False)
def get_llama_matrix_scores_cached(patient_id, user_data_json):
    tokenizer, model, device = load_llama_model()

    user_data = json.loads(user_data_json)

    prompt = build_exercise_matrix_llama_prompt(user_data)

    messages = [
        {
            "role": "system",
            "content": "You are a JSON-only scoring assistant for a rehabilitation dashboard."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=220,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    parsed = extract_json_object(response)
    validated_df = validate_llama_matrix_scores(parsed)

    if validated_df is None:
        raise ValueError(f"Llama returned invalid matrix JSON: {response}")

    return validated_df


def get_radar_big5_scores(user_data):
    """
    Uses Llama for the radar chart if enabled.
    Falls back to rule-based Big Five scoring if Llama fails.
    """

    if not USE_LLAMA_FOR_RADAR:
        return get_big5_scores(user_data), "Rule-based"

    patient_id = get_patient_id(user_data)
    user_data_json = json.dumps(user_data, ensure_ascii=False, sort_keys=True)

    try:
        llama_scores = get_llama_big5_scores_cached(patient_id, user_data_json)
        return llama_scores, "Llama 3.1 8B"
    except Exception as e:
        st.warning(f"Llama scoring failed. Falling back to rule-based scoring. Error: {e}")
        return get_big5_scores(user_data), "Rule-based fallback"

def get_matrix_scores(user_data):
    """
    Uses Llama for the exercise suitability matrix if enabled.
    Falls back to rule-based matrix scoring if Llama fails.
    """

    if not USE_LLAMA_FOR_MATRIX:
        return get_exercise_suitability_matrix_df_rule_based(user_data), "Rule-based"

    patient_id = get_patient_id(user_data)
    user_data_json = json.dumps(user_data, ensure_ascii=False, sort_keys=True)

    try:
        matrix_df = get_llama_matrix_scores_cached(patient_id, user_data_json)
        return matrix_df, "Llama 3.1 8B"
    except Exception as e:
        st.warning(f"Llama matrix scoring failed. Falling back to rule-based scoring. Error: {e}")
        return get_exercise_suitability_matrix_df_rule_based(user_data), "Rule-based fallback"

# --------------------------------------------------
# Reusable UI components
# --------------------------------------------------
def metric_card(label, value):
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def chart_placeholder(title, description, height="normal"):
    box_class = "placeholder-box" if height == "normal" else "placeholder-box-small"

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="{box_class}">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_factor_pill(label, value, style="neutral"):
    return {
        "label": str(label),
        "value": translate_german_value(value),
        "style": style
    }


def render_factor_pills(pills):
    """
    Renders pill data as HTML.
    This expects pills to be dictionaries returned by make_factor_pill().
    """

    style_map = {
        "good": {
            "bg": "#DCFCE7",
            "color": "#166534",
            "border": "#BBF7D0"
        },
        "warning": {
            "bg": "#FEF3C7",
            "color": "#92400E",
            "border": "#FDE68A"
        },
        "risk": {
            "bg": "#FEE2E2",
            "color": "#991B1B",
            "border": "#FECACA"
        },
        "neutral": {
            "bg": "#E0F2FE",
            "color": "#075985",
            "border": "#BAE6FD"
        }
    }

    html = """
    <div style="
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 6px;
        margin-bottom: 8px;
    ">
    """

    for pill in pills:
        # Safety fallback in case an old HTML string still sneaks in
        if not isinstance(pill, dict):
            continue

        style = pill.get("style", "neutral")
        chosen = style_map.get(style, style_map["neutral"])

        label = escape(str(pill.get("label", "")))
        value = escape(str(pill.get("value", "")))

        html += f"""
        <span style="
            display: inline-flex;
            align-items: center;
            padding: 8px 12px;
            border-radius: 999px;
            background-color: {chosen["bg"]};
            color: {chosen["color"]};
            border: 1px solid {chosen["border"]};
            font-size: 13px;
            font-weight: 700;
            line-height: 1.2;
            white-space: nowrap;
        ">
            {label}: {value}
        </span>
        """

    html += "</div>"

    # Use st.html if available; otherwise fallback to st.markdown
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)



# --------------------------------------------------
# Exercise difficulty data preparation
# --------------------------------------------------
def get_perceived_actual_difficulty_df(user_data):
    """
    Builds a dataframe for perceived vs actual difficulty.

    Supports two formats:
    1. Raw nested JSON record
    2. Flattened user_data dictionary
    """

    rows = []

    # --------------------------------------------------
    # Case 1: raw nested JSON record
    # --------------------------------------------------
    if "exercise_task" in user_data:
        exercise_task = user_data.get("exercise_task", {})
        additional_tasks = user_data.get("additional_exercise_tasks", {})

        exercise_map = {
            "Squats": exercise_task,
            "Calf Raises": additional_tasks.get("calf_raises", {}),
            "Standing Toe Touch": additional_tasks.get("standing_toe_touches", {}),
            "Toe Touch Stretch": additional_tasks.get("toe_touch_stretches", {}),
        }

        for exercise_name, task_data in exercise_map.items():
            perceived_score = extract_numeric_score(
                task_data.get("perceived_difficulty", "-")
            )

            actual_score = extract_numeric_score(
                task_data.get("actual_difficulty", "-")
            )

            if perceived_score > 0:
                rows.append({
                    "Exercise": exercise_name,
                    "Difficulty Type": "Perceived",
                    "Difficulty Score": perceived_score
                })

            if actual_score > 0:
                rows.append({
                    "Exercise": exercise_name,
                    "Difficulty Type": "Actual",
                    "Difficulty Score": actual_score
                })

    # --------------------------------------------------
    # Case 2: flattened user_data from record_to_user_data()
    # --------------------------------------------------
    elif "question_answer_string" in user_data:
        exercise_map = {
            "Squats": (
                user_data.get("question_answer_string", 0),
                user_data.get("actual_squat_score", 0)
            ),
            "Calf Raises": (
                user_data.get("question_answer_string2", 0),
                user_data.get("actual_calf_score", 0)
            ),
            "Standing Toe Touch": (
                user_data.get("question_answer_string3", 0),
                user_data.get("actual_toe_touch_score", 0)
            ),
            "Toe Touch Stretch": (
                user_data.get("question_answer_string4", 0),
                user_data.get("actual_toe_touch_stretch_score", 0)
            ),
        }

        for exercise_name, (perceived_score, actual_score) in exercise_map.items():
            perceived_score = int(perceived_score or 0)
            actual_score = int(actual_score or 0)

            if perceived_score > 0:
                rows.append({
                    "Exercise": exercise_name,
                    "Difficulty Type": "Perceived",
                    "Difficulty Score": perceived_score
                })

            if actual_score > 0:
                rows.append({
                    "Exercise": exercise_name,
                    "Difficulty Type": "Actual",
                    "Difficulty Score": actual_score
                })

    # --------------------------------------------------
    # Case 3: old mock fallback
    # --------------------------------------------------
    else:
        perceived_score = int(user_data.get("Perceived_Difficulty", 0) or 0)
        actual_score = int(user_data.get("Actual_Difficulty", 0) or 0)

        rows = [
            {
                "Exercise": "Selected Exercise",
                "Difficulty Type": "Perceived",
                "Difficulty Score": perceived_score
            },
            {
                "Exercise": "Selected Exercise",
                "Difficulty Type": "Actual",
                "Difficulty Score": actual_score
            },
        ]

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df[df["Difficulty Score"] > 0]

    return df


def get_exercise_scores_for_comparison(user_data):
    """
    Extracts exercise perceived and actual difficulty scores from the raw nested JSON.
    Returns one row per exercise.
    """

    exercise_task = user_data.get("exercise_task", {})
    additional_tasks = user_data.get("additional_exercise_tasks", {})

    exercise_map = {
        "Squats": exercise_task,
        "Calf Raises": additional_tasks.get("calf_raises", {}),
        "Standing Toe Touch": additional_tasks.get("standing_toe_touches", {}),
        "Toe Touch Stretch": additional_tasks.get("toe_touch_stretches", {}),
    }

    rows = []

    for exercise_name, task_data in exercise_map.items():
        perceived_score = extract_numeric_score(
            task_data.get("perceived_difficulty", "-")
        )
        actual_score = extract_numeric_score(
            task_data.get("actual_difficulty", "-")
        )

        if perceived_score > 0 or actual_score > 0:
            rows.append({
                "Exercise": exercise_name,
                "Perceived": perceived_score,
                "Actual": actual_score,
                "Gap": actual_score - perceived_score
            })

    return pd.DataFrame(rows)


def get_biggest_exercise_difference(user_data):
    """
    Finds the two exercises with the largest actual difficulty difference.
    Returns easier exercise, harder exercise, and score difference.
    """

    df = get_exercise_scores_for_comparison(user_data)

    if df.empty or len(df) < 2:
        return None, None, 0, df

    df = df[df["Actual"] > 0]

    if len(df) < 2:
        return None, None, 0, df

    easiest = df.loc[df["Actual"].idxmin()].to_dict()
    hardest = df.loc[df["Actual"].idxmax()].to_dict()

    difference = hardest["Actual"] - easiest["Actual"]

    return easiest, hardest, difference, df



# --------------------------------------------------
# Exercise comparison explanation pills
# --------------------------------------------------
def get_patient_factor_pills(user_data):
    """
    Creates only high-value patient-context pills.
    Avoids clutter by not showing normal/negative findings.
    """

    health = user_data.get("health", {})
    demographics = user_data.get("demographics", {})
    lifestyle = user_data.get("lifestyle", {})
    psychology = user_data.get("psychology", {})

    pills = []

    age = demographics.get("age", "-")
    disability = health.get("disability", "-")
    overall_health = health.get("overall_health", "-")
    mobility = health.get("mobility", "-")
    surgery = health.get("surgery", "-")
    surgery_complications = health.get("surgery_complications", "-")
    recovery = health.get("recovery", "-")
    pt_adherence = health.get("pt_adherence", "-")

    days_per_week = lifestyle.get("days_per_week", "-")
    stress_24h = psychology.get("stress_or_anxiety_24h", "-")

    age_score = extract_numeric_score(age)
    health_score = extract_numeric_score(overall_health)
    mobility_score = extract_numeric_score(mobility)
    pt_score = extract_numeric_score(pt_adherence)

    # Age only if older group; avoid showing age for everyone
    if age_score >= 55:
        pills.append(make_factor_pill("Age factor", age, "warning"))

    # Surgery only if yes
    if yes_no_value(surgery):
        pills.append(make_factor_pill("Recent surgery", "Yes", "warning"))

        if recovery != "-":
            pills.append(make_factor_pill("Recovery time", recovery, "warning"))

    # Complications only if yes
    if yes_no_value(surgery_complications):
        pills.append(make_factor_pill("Surgery complications", "Yes", "risk"))

    # Disability only if yes
    if yes_no_value(disability):
        pills.append(make_factor_pill("Physical disability", "Yes", "risk"))

    # Health only if not good
    if health_score > 0 and health_score <= 3:
        pills.append(make_factor_pill("Overall health", f"{health_score}/5", "warning"))

    # Mobility only if not good
    if mobility_score > 0 and mobility_score <= 3:
        pills.append(make_factor_pill("Mobility", f"{mobility_score}/5", "warning"))

    # Low exercise frequency only
    if "0" in str(days_per_week) or "1-2" in str(days_per_week):
        pills.append(
            make_factor_pill(
                "Exercise frequency",
                translate_german_value(days_per_week),
                "warning"
            )
        )

    # PT adherence only if not high
    if pt_score > 0 and pt_score <= 3:
        pills.append(make_factor_pill("PT adherence", f"{pt_score}/5", "warning"))

    # Stress only if yes
    if yes_no_value(stress_24h):
        pills.append(make_factor_pill("Recent stress/anxiety", "Yes", "warning"))

    return pills


def get_exercise_specific_pills(easier, harder, user_data):
    """
    Creates a small, prioritized set of exercise-comparison pills.
    Only shows factors that help explain the difference.
    """

    health = user_data.get("health", {})
    lifestyle = user_data.get("lifestyle", {})

    surgery = yes_no_value(health.get("surgery", "-"))
    disability = yes_no_value(health.get("disability", "-"))
    mobility_score = extract_numeric_score(health.get("mobility", "-"))
    days_per_week = lifestyle.get("days_per_week", "-")

    easier_name = easier["Exercise"]
    harder_name = harder["Exercise"]

    pills = []

    # 1. Always show actual difficulty gap
    pills.append(
        make_factor_pill(
            "Difficulty gap",
            f'{harder_name} {harder["Actual"]}/5 vs {easier_name} {easier["Actual"]}/5',
            "warning"
        )
    )

    # 2. Exercise movement demand
    lower_limb_load_exercises = {"Squats", "Calf Raises"}
    flexibility_exercises = {"Standing Toe Touch", "Toe Touch Stretch"}

    if harder_name in lower_limb_load_exercises:
        pills.append(
            make_factor_pill(
                "Exercise demand",
                f"{harder_name} may require repeated lower-limb loading",
                "warning"
            )
        )

    elif harder_name in flexibility_exercises:
        pills.append(
            make_factor_pill(
                "Exercise demand",
                f"{harder_name} may require more flexibility or trunk movement",
                "warning"
            )
        )

    if easier_name in flexibility_exercises:
        pills.append(
            make_factor_pill(
                "Easier movement pattern",
                f"{easier_name} may involve less repeated loading",
                "good"
            )
        )

    # 3. Surgery context only if surgery is Yes
    if surgery:
        pills.append(
            make_factor_pill(
                "Surgery history",
                "Yes",
                "warning"
            )
        )

    # 4. Disability only if present
    if disability:
        pills.append(
            make_factor_pill(
                "Physical disability",
                "Yes",
                "risk"
            )
        )

    # 5. Mobility only if low/moderate
    if mobility_score > 0 and mobility_score <= 3:
        pills.append(
            make_factor_pill(
                "Mobility",
                f"{mobility_score}/5",
                "warning"
            )
        )

    # 6. Low exercise habit only
    if "0" in str(days_per_week) or "1-2" in str(days_per_week):
        pills.append(
            make_factor_pill(
                "Exercise frequency",
                translate_german_value(days_per_week),
                "warning"
            )
        )

    # 7. Large mismatch only, not every mismatch
    easier_gap = abs(easier["Actual"] - easier["Perceived"])
    harder_gap = abs(harder["Actual"] - harder["Perceived"])

    if harder_gap >= 2:
        pills.append(
            make_factor_pill(
                f"{harder_name} mismatch",
                f'actual {harder["Actual"]}/5 vs perceived {harder["Perceived"]}/5',
                "warning"
            )
        )

    elif easier_gap >= 2:
        pills.append(
            make_factor_pill(
                f"{easier_name} mismatch",
                f'actual {easier["Actual"]}/5 vs perceived {easier["Perceived"]}/5',
                "warning"
            )
        )

    return pills


def select_top_factor_pills(exercise_pills, patient_pills, max_pills=5):
    """
    Keeps the explanation short.
    Prioritizes exercise-specific comparison reasons first,
    then patient-context reasons.
    """

    combined = exercise_pills + patient_pills

    # Remove duplicates based on label + value
    seen = set()
    unique = []

    for pill in combined:
        key = (pill.get("label"), pill.get("value"))

        if key not in seen:
            unique.append(pill)
            seen.add(key)

    return unique[:max_pills]



# --------------------------------------------------
# Exercise suitability matrix helpers
# --------------------------------------------------
def score_to_color(score):
    """
    Preserves the matrix color scheme:
    1 = green
    2 = light green
    3 = yellow
    4 = orange
    5 = red
    """

    color_map = {
        1: {
            "bg": "linear-gradient(135deg, #4ADE80, #22C55E)",
            "text": "#FFFFFF"
        },
        2: {
            "bg": "linear-gradient(135deg, #BEF264, #A3E635)",
            "text": "#0F172A"
        },
        3: {
            "bg": "linear-gradient(135deg, #FDE047, #FACC15)",
            "text": "#0F172A"
        },
        4: {
            "bg": "linear-gradient(135deg, #FB923C, #F97316)",
            "text": "#0F172A"
        },
        5: {
            "bg": "linear-gradient(135deg, #F43F5E, #EF4444)",
            "text": "#FFFFFF"
        }
    }

    return color_map.get(score, color_map[3])


def get_activity_score(days_per_week):
    text = str(days_per_week).lower()

    if "7" in text:
        return 5
    if "5-6" in text:
        return 4
    if "3-4" in text:
        return 3
    if "1-2" in text:
        return 2
    if "0" in text:
        return 1

    return 3


def get_exercise_base_type(exercise_name):
    lower_limb_load = {"Squats", "Calf Raises"}
    flexibility = {"Standing Toe Touch", "Toe Touch Stretch"}

    if exercise_name in lower_limb_load:
        return "load"
    if exercise_name in flexibility:
        return "flexibility"

    return "general"

MATRIX_COLUMNS = [
    "Predicted Difficulty",
    "Pain Risk",
    "Readiness Match",
    "Support Need",
]


def build_exercise_matrix_llama_prompt(user_data):
    health = user_data.get("health", {})
    lifestyle = user_data.get("lifestyle", {})
    exercise_task = user_data.get("exercise_task", {})
    additional_tasks = user_data.get("additional_exercise_tasks", {})

    exercise_summary = {
        "patient_context": {
            "overall_health": health.get("overall_health", "-"),
            "mobility": health.get("mobility", "-"),
            "disability": health.get("disability", "-"),
            "surgery": health.get("surgery", "-"),
            "surgery_complications": health.get("surgery_complications", "-"),
            "recovery": health.get("recovery", "-"),
            "pt_after": health.get("pt_after", "-"),
            "pt_adherence": health.get("pt_adherence", "-"),
            "daily_activities_without_difficulty": health.get("daily_activities_without_difficulty", "-"),
            "activities": lifestyle.get("activities", "-"),
            "days_per_week": lifestyle.get("days_per_week", "-"),
            "session_length": lifestyle.get("session_length", "-"),
        },
        "exercise_scores": {
            "Squats": exercise_task,
            "Calf Raises": additional_tasks.get("calf_raises", {}),
            "Standing Toe Touch": additional_tasks.get("standing_toe_touches", {}),
            "Toe Touch Stretch": additional_tasks.get("toe_touch_stretches", {}),
        }
    }

    return f"""
You are scoring a rehabilitation dashboard exercise suitability matrix.

Task:
Given structured patient survey data and exercise difficulty scores, assign four scores for each exercise.

Exercises:
- Squats
- Calf Raises
- Standing Toe Touch
- Toe Touch Stretch

Matrix columns:
- Predicted Difficulty
- Pain Risk
- Readiness Match
- Support Need

Scoring rules:
- Return ONLY valid JSON.
- Do not include explanations.
- Do not diagnose the patient.
- Do not give medical advice.
- Each score must be an integer from 1 to 5.
- For Predicted Difficulty, 1 means very low difficulty and 5 means very high difficulty.
- For Pain Risk, 1 means low concern and 5 means high concern.
- For Readiness Match, 1 means poor match and 5 means strong match.
- For Support Need, 1 means low support need and 5 means high support need.
- Use the provided perceived and actual difficulty scores.
- Also consider mobility, surgery history, complications, disability, activity level, and PT adherence.

Patient and exercise data:
{json.dumps(exercise_summary, ensure_ascii=False, indent=2)}

Required JSON format:
{{
  "Squats": {{
    "Predicted Difficulty": 0,
    "Pain Risk": 0,
    "Readiness Match": 0,
    "Support Need": 0
  }},
  "Calf Raises": {{
    "Predicted Difficulty": 0,
    "Pain Risk": 0,
    "Readiness Match": 0,
    "Support Need": 0
  }},
  "Standing Toe Touch": {{
    "Predicted Difficulty": 0,
    "Pain Risk": 0,
    "Readiness Match": 0,
    "Support Need": 0
  }},
  "Toe Touch Stretch": {{
    "Predicted Difficulty": 0,
    "Pain Risk": 0,
    "Readiness Match": 0,
    "Support Need": 0
  }}
}}
""".strip()

def get_exercise_suitability_matrix_df_rule_based(user_data):
    """
    Builds the exercise suitability matrix from patient JSON.

    Scores:
    1 = low
    5 = high

    For Readiness Match:
    1 = poor match
    5 = strong match
    """

    health = user_data.get("health", {})
    lifestyle = user_data.get("lifestyle", {})

    overall_health_score = extract_numeric_score(
        health.get("overall_health", "-")
    )
    mobility_score = extract_numeric_score(
        health.get("mobility", "-")
    )
    activity_score = get_activity_score(
        lifestyle.get("days_per_week", "-")
    )

    if overall_health_score == 0:
        overall_health_score = 3

    if mobility_score == 0:
        mobility_score = 3

    had_surgery = yes_no_value(
        health.get("surgery", "-")
    )
    had_complications = yes_no_value(
        health.get("surgery_complications", "-")
    )
    has_disability = yes_no_value(
        health.get("disability", "-")
    )

    exercise_task = user_data.get("exercise_task", {})
    additional_tasks = user_data.get("additional_exercise_tasks", {})

    exercise_map = {
        "Squats": exercise_task,
        "Calf Raises": additional_tasks.get("calf_raises", {}),
        "Standing Toe Touch": additional_tasks.get("standing_toe_touches", {}),
        "Toe Touch Stretch": additional_tasks.get("toe_touch_stretches", {}),
    }

    rows = []

    for exercise_name, task_data in exercise_map.items():
        perceived = extract_numeric_score(
            task_data.get("perceived_difficulty", "-")
        )
        actual = extract_numeric_score(
            task_data.get("actual_difficulty", "-")
        )

        if perceived == 0 and actual == 0:
            continue

        # Use actual difficulty if present, otherwise perceived.
        predicted_difficulty = actual if actual > 0 else perceived
        predicted_difficulty = clamp_score(predicted_difficulty)

        exercise_type = get_exercise_base_type(exercise_name)

        # -----------------------------
        # Pain risk
        # -----------------------------
        pain_risk = predicted_difficulty

        if had_surgery and exercise_type == "load":
            pain_risk += 1

        if had_complications:
            pain_risk += 1

        if has_disability:
            pain_risk += 1

        if mobility_score <= 3 and exercise_type in ["load", "flexibility"]:
            pain_risk += 1

        if overall_health_score <= 3:
            pain_risk += 1

        pain_risk = clamp_score(pain_risk)

        # -----------------------------
        # Readiness match
        # Higher = better match
        # -----------------------------
        readiness_base = round(
            (overall_health_score + mobility_score + activity_score) / 3
        )

        readiness_match = readiness_base

        if predicted_difficulty >= 4:
            readiness_match -= 1

        if pain_risk >= 4:
            readiness_match -= 1

        if exercise_type == "load" and had_surgery:
            readiness_match -= 1

        readiness_match = clamp_score(readiness_match)

        # -----------------------------
        # Support need
        # Higher = more support needed
        # -----------------------------
        support_need = round(
            (predicted_difficulty + pain_risk + (6 - readiness_match)) / 3
        )

        if had_surgery and exercise_type == "load":
            support_need += 1

        if has_disability:
            support_need += 1

        support_need = clamp_score(support_need)

        rows.append({
            "Exercise": exercise_name,
            "Predicted Difficulty": predicted_difficulty,
            "Pain Risk": pain_risk,
            "Readiness Match": readiness_match,
            "Support Need": support_need
        })

    return pd.DataFrame(rows)

def get_exercise_suitability_matrix_df(user_data):
    matrix_df, _ = get_matrix_scores(user_data)
    return matrix_df

# --------------------------------------------------
# Chart renderers
# --------------------------------------------------
def render_icf_bps_domain_profile(user_data, big5_scores=None, score_source="Rule-based"):
    domain_df = get_icf_bps_domain_profile_df(user_data, big5_scores)

    with st.container(border=True):
        st.markdown(
            """
            <div style="
                font-size: 24px;
                font-weight: 850;
                color: #0B2E59;
                margin-bottom: 8px;
            ">
                ICF Psychological Domain Profile
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Big Five personality scores mapped to structured rehabilitation-relevant domains.
            </div>
            """,
            unsafe_allow_html=True
        )

        if domain_df.empty:
            st.info("No personality trait data available for this patient.")
            return

        chart = (
            alt.Chart(domain_df)
            .mark_bar(
                size=18,
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            )
            .encode(
                y=alt.Y(
                    "Domain:N",
                    sort=[
                        "Emotional regulation",
                        "Activity self-management",
                        "Participation engagement",
                        "Social cooperation / support",
                        "Adaptability to rehabilitation"
                    ],
                    title=None,
                    axis=alt.Axis(
                        labelFontSize=12,
                        labelColor="#475569",
                        labelLimit=190,
                        tickSize=0,
                        domain=False
                    )
                ),
                x=alt.X(
                    "Score:Q",
                    title="Score out of 7",
                    scale=alt.Scale(domain=[0, 7]),
                    axis=alt.Axis(
                        values=[0, 1, 2, 3, 4, 5, 6, 7],
                        labelFontSize=12,
                        labelColor="#64748B",
                        titleFontSize=13,
                        titleColor="#64748B",
                        grid=True,
                        gridColor="#E2E8F0",
                        domain=False,
                        tickSize=0
                    )
                ),
                yOffset=alt.YOffset(
                    "Legend:N",
                    sort=["Patient", "General Norm"]
                ),
                color=alt.Color(
                    "Legend:N",
                    scale=alt.Scale(
                        domain=["Patient", "General Norm"],
                        range=["#38BDF8", "#F59E0B"]
                    ),
                    legend=alt.Legend(
                        title="Profile",
                        orient="top",
                        titleFontSize=13,
                        labelFontSize=12,
                        symbolType="square"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Domain:N", title="ICF/BPS domain"),
                    alt.Tooltip("Trait:N", title="Mapped trait"),
                    alt.Tooltip("Legend:N", title="Profile"),
                    alt.Tooltip("Score:Q", title="Score", format=".2f")
                ]
            )
            .properties(height=340)
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div style="
                font-size: 12px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 8px;
            ">
                This visualization does not diagnose psychological state. It displays survey-derived personality traits
                as structured AMM dimensions for clinical inspection.
            </div>
            """,
            unsafe_allow_html=True
        )


def render_personality_trait_radar(user_data, big5_scores=None, score_source=None):
    if big5_scores is None or score_source is None:
        big5_scores, score_source = get_radar_big5_scores(user_data)

    traits = [
        "Agreeableness",
        "Conscientiousness",
        "Extraversion",
        "Emotional Stability",
        "Openness"
    ]

    patient_values = [big5_scores.get(trait, 0) for trait in traits]
    norm_values = [GENERAL_NORMS.get(trait, 0) for trait in traits]

    labels = [
        "Social support\nAgreeableness",
        "Self-management\nConscientiousness",
        "Participation\nExtraversion",
        "Emotion regulation\nEmotional Stability",
        "Adaptability\nOpenness"
    ]

    # Close the radar loop
    patient_values += patient_values[:1]
    norm_values += norm_values[:1]
    labels += labels[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    with st.container(border=True):
        st.markdown(
            """
            <div style="
                font-size: 24px;
                font-weight: 850;
                color: #0B2E59;
                margin-bottom: 8px;
            ">
                Personality Trait Radar
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Patient Big Five profile compared with general norm values. Patient scores generated by Llama 3.1 8B.
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))

        ax.plot(angles, patient_values, linewidth=2, label="Patient")
        ax.fill(angles, patient_values, alpha=0.18)

        ax.plot(angles, norm_values, linewidth=2, linestyle="--", label="General Norm")
        ax.fill(angles, norm_values, alpha=0.08)

        ax.set_ylim(0, 7)
        ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
        ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"], fontsize=8, color="#64748B")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels[:-1], fontsize=9, color="#0B2E59")

        ax.tick_params(pad=10)
        ax.grid(color="#E2E8F0")
        ax.spines["polar"].set_color("#CBD5E1")

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.25, 1.12),
            frameon=False,
            fontsize=9
        )

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        st.pyplot(fig, use_container_width=True)

        st.markdown(
            """
            <div style="
                font-size: 12px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 8px;
            ">
                Radar axes show Big Five traits with their mapped AMM/ICF-style interpretation.
                Scores are shown on a 1–7 scale.
            </div>
            """,
            unsafe_allow_html=True
        )


def render_perceived_actual_chart(user_data):
    difficulty_df = get_perceived_actual_difficulty_df(user_data)

    with st.container(border=True):
        st.markdown(
            """
            <div style="
                font-size: 24px;
                font-weight: 800;
                color: #0B2E59;
                margin-bottom: 8px;
            ">
                Perceived vs Actual Difficulty
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Comparison of expected and actual exercise difficulty on a 1–5 scale.
            </div>
            """,
            unsafe_allow_html=True
        )

        if difficulty_df.empty:
            st.info("No perceived/actual difficulty scores available for this patient.")
            return

        chart = (
            alt.Chart(difficulty_df)
            .mark_bar(
                size=26,
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            )
            .encode(
                x=alt.X(
                    "Exercise:N",
                    sort=[
                        "Squats",
                        "Calf Raises",
                        "Standing Toe Touch",
                        "Toe Touch Stretch"
                    ],
                    title=None,
                    axis=alt.Axis(
                        labelAngle=-15,
                        labelFontSize=12,
                        labelColor="#475569",
                        labelLimit=130,
                        tickSize=0,
                        domain=False
                    )
                ),
                xOffset=alt.XOffset(
                    "Difficulty Type:N",
                    sort=["Perceived", "Actual"]
                ),
                y=alt.Y(
                    "Difficulty Score:Q",
                    title="Difficulty score",
                    scale=alt.Scale(domain=[0, 5]),
                    axis=alt.Axis(
                        values=[0, 1, 2, 3, 4, 5],
                        labelFontSize=12,
                        labelColor="#64748B",
                        titleFontSize=13,
                        titleColor="#64748B",
                        grid=True,
                        gridColor="#E2E8F0",
                        domain=False,
                        tickSize=0
                    )
                ),
                color=alt.Color(
                    "Difficulty Type:N",
                    scale=alt.Scale(
                        domain=["Perceived", "Actual"],
                        range=["#38BDF8", "#F59E0B"]
                    ),
                    legend=alt.Legend(
                        title="Difficulty type",
                        titleFontSize=13,
                        labelFontSize=12,
                        orient="top-right",
                        symbolType="square"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Exercise:N", title="Exercise"),
                    alt.Tooltip("Difficulty Type:N", title="Type"),
                    alt.Tooltip("Difficulty Score:Q", title="Score", format=".0f")
                ]
            )
            .properties(
                height=310,
                padding={"left": 10, "right": 20, "top": 10, "bottom": 10}
            )
            .configure_view(
                strokeWidth=0
            )
            .configure_axis(
                labelFont="Arial",
                titleFont="Arial"
            )
            .configure_legend(
                labelFont="Arial",
                titleFont="Arial",
                padding=8
            )
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div style="
                font-size: 13px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 6px;
            ">
                Higher values indicate greater difficulty. A gap between perceived and actual difficulty
                may indicate overestimation or underestimation of exercise effort.
            </div>
            """,
            unsafe_allow_html=True
        )


def render_exercise_comparison_card(user_data):
    easier, harder, difference, df = get_biggest_exercise_difference(user_data)

    with st.container(border=True):
        st.markdown(
            """
            <div style="
                font-size: 24px;
                font-weight: 800;
                color: #0B2E59;
                margin-bottom: 8px;
            ">
                Exercise Comparison
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Compares the two exercises with the largest actual difficulty difference.
            </div>
            """,
            unsafe_allow_html=True
        )

        if easier is None or harder is None or difference == 0:
            st.info("No meaningful exercise difficulty difference found for this patient.")
            return

        comparison_df = pd.DataFrame([
            {
                "Exercise": easier["Exercise"],
                "Actual Difficulty": easier["Actual"],
                "Role": "Easier"
            },
            {
                "Exercise": harder["Exercise"],
                "Actual Difficulty": harder["Actual"],
                "Role": "Harder"
            }
        ])

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-radius: 16px;
                    padding: 16px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 13px; color: #64748B; font-weight: 700;">
                        Easier exercise
                    </div>
                    <div style="font-size: 22px; color: #166534; font-weight: 850;">
                        {easier["Exercise"]}
                    </div>
                    <div style="font-size: 14px; color: #475569; margin-top: 4px;">
                        Actual difficulty: <b>{easier["Actual"]}/5</b> · Perceived: <b>{easier["Perceived"]}/5</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #F8FAFC;
                    border: 1px solid #E2E8F0;
                    border-radius: 16px;
                    padding: 16px;
                    margin-bottom: 10px;
                ">
                    <div style="font-size: 13px; color: #64748B; font-weight: 700;">
                        Harder exercise
                    </div>
                    <div style="font-size: 22px; color: #991B1B; font-weight: 850;">
                        {harder["Exercise"]}
                    </div>
                    <div style="font-size: 14px; color: #475569; margin-top: 4px;">
                        Actual difficulty: <b>{harder["Actual"]}/5</b> · Perceived: <b>{harder["Perceived"]}/5</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        chart = (
            alt.Chart(comparison_df)
            .mark_bar(
                size=45,
                cornerRadiusTopLeft=6,
                cornerRadiusTopRight=6
            )
            .encode(
                x=alt.X(
                    "Exercise:N",
                    title=None,
                    sort=[easier["Exercise"], harder["Exercise"]],
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=12,
                        labelColor="#475569",
                        domain=False,
                        tickSize=0
                    )
                ),
                y=alt.Y(
                    "Actual Difficulty:Q",
                    title="Actual difficulty",
                    scale=alt.Scale(domain=[0, 5]),
                    axis=alt.Axis(
                        values=[0, 1, 2, 3, 4, 5],
                        labelFontSize=12,
                        titleFontSize=13,
                        labelColor="#64748B",
                        titleColor="#64748B",
                        grid=True,
                        gridColor="#E2E8F0",
                        domain=False,
                        tickSize=0
                    )
                ),
                color=alt.Color(
                    "Role:N",
                    scale=alt.Scale(
                        domain=["Easier", "Harder"],
                        range=["#22C55E", "#EF4444"]
                    ),
                    legend=None
                ),
                tooltip=[
                    alt.Tooltip("Exercise:N", title="Exercise"),
                    alt.Tooltip("Role:N", title="Role"),
                    alt.Tooltip("Actual Difficulty:Q", title="Actual Difficulty", format=".0f")
                ]
            )
            .properties(height=220)
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div style="
                font-size: 15px;
                font-weight: 800;
                color: #0B2E59;
                margin-top: 8px;
                margin-bottom: 8px;
            ">
                Why this difference may occur?
            </div>
            """,
            unsafe_allow_html=True
        )

        exercise_pills = get_exercise_specific_pills(easier, harder, user_data)
        patient_pills = get_patient_factor_pills(user_data)

        all_pills = select_top_factor_pills(
            exercise_pills,
            patient_pills,
            max_pills=5
        )

        render_factor_pills(all_pills)

        st.markdown(
            """
            <div style="
                font-size: 12px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 10px;
            ">
            </div>
            """,
            unsafe_allow_html=True
        )


def render_exercise_suitability_matrix(user_data, matrix_df=None, score_source="Rule-based"):
    if matrix_df is None:
        matrix_df, score_source = get_matrix_scores(user_data)

    exercise_icons = {
        "Squats": "🏋️",
        "Calf Raises": "🦵",
        "Standing Toe Touch": "🤸",
        "Toe Touch Stretch": "🧘"
    }

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 8px;
            ">
                <div style="
                    font-size: 26px;
                    color: #2563EB;
                ">
                    ▦
                </div>
                <div style="
                    font-size: 24px;
                    font-weight: 850;
                    color: #0B2E59;
                ">
                    Exercise Suitability Matrix
                </div>
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Matrix scores generated using: <b>{score_source}</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

        if matrix_df.empty:
            st.info("No exercise suitability data available for this patient.")
            return

        columns = [
            "Predicted Difficulty",
            "Pain Risk",
            "Readiness Match",
            "Support Need"
        ]

        rows_html = ""

        for _, row in matrix_df.iterrows():
            exercise = str(row["Exercise"])
            icon = exercise_icons.get(exercise, "🏃")

            rows_html += f"""
            <tr>
                <td class="exercise-cell">
                    <span class="exercise-icon">{icon}</span>
                    {escape(exercise)}
                </td>
            """

            for col in columns:
                score = int(row[col])
                colors = score_to_color(score)

                rows_html += f"""
                <td class="score-cell" style="
                    background: {colors["bg"]};
                    color: {colors["text"]};
                ">
                    {score}
                </td>
                """

            rows_html += "</tr>"

        header_html = ""

        for col in columns:
            header_html += f"<th>{escape(col)}</th>"

        html = f"""
        <style>
            .matrix-wrapper {{
                width: 100%;
                overflow-x: auto;
                margin-top: 4px;
            }}

            .suitability-table {{
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                font-family: Arial, sans-serif;
                border: 1px solid #E2E8F0;
                border-radius: 14px;
                overflow: hidden;
                background-color: white;
            }}

            .suitability-table th {{
                padding: 14px 18px;
                text-align: center;
                color: #0B2E59;
                font-size: 15px;
                font-weight: 800;
                background-color: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
                white-space: nowrap;
            }}

            .suitability-table th:first-child {{
                width: 260px;
                text-align: left;
            }}

            .exercise-cell {{
                padding: 18px 20px;
                background-color: #FFFFFF;
                border-bottom: 1px solid #E2E8F0;
                color: #0B2E59;
                font-size: 18px;
                font-weight: 750;
                white-space: nowrap;
            }}

            .exercise-icon {{
                display: inline-block;
                width: 34px;
                font-size: 24px;
                margin-right: 12px;
                vertical-align: middle;
            }}

            .score-cell {{
                text-align: center;
                font-size: 22px;
                font-weight: 850;
                height: 64px;
                border-bottom: 1px solid #FFFFFF;
                border-left: 1px solid #FFFFFF;
                min-width: 130px;
            }}

            .matrix-note {{
                font-size: 12px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 12px;
            }}
        </style>

        <div class="matrix-wrapper">
            <table class="suitability-table">
                <thead>
                    <tr>
                        <th></th>
                        {header_html}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>

        <br><br><br>
        <div class="matrix-descriptions">
            <div class="description-item">
                <b>Predicted Difficulty:</b> Estimated exercise difficulty based on the patient’s reported perceived and actual difficulty scores.
            </div>
            <div class="description-item">
                <b>Pain Risk:</b> Estimated risk of discomfort based on difficulty, surgery history, disability, mobility, and overall health.
            </div>
            <div class="description-item">
                <b>Readiness Match:</b> Shows how well the exercise matches the patient’s current health, mobility, and activity level.
            </div>
            <div class="description-item">
                <b>Support Need:</b> Indicates how much supervision or rehabilitation support may be needed for the exercise.
            </div>
        </div>

        <div class="matrix-note">
            Scores are derived from structured patient data. Green indicates lower concern,
            yellow indicates moderate concern, and red indicates higher concern.
            For Readiness Match, higher values indicate a better match between the patient profile and the exercise.
        </div>
        """

        st.html(textwrap.dedent(html).strip())


def render_hardcoded_pain_forecast():
    """
    Hardcoded 12-month pain/difficulty forecast.
    This is only for UI/demo purposes, not a clinical prediction.
    """

    forecast_df = pd.DataFrame({
        "Month": list(range(1, 13)),
        "Pain Score": [4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1]
    })

    with st.container(border=True):
        st.markdown(
            """
            <div style="
                font-size: 24px;
                font-weight: 850;
                color: #0B2E59;
                margin-bottom: 8px;
            ">
                12-Month Pain Forecast
            </div>
            <div style="
                font-size: 14px;
                color: #64748B;
                margin-bottom: 18px;
            ">
                Illustrative monthly pain/difficulty trajectory on a 1–5 scale.
            </div>
            """,
            unsafe_allow_html=True
        )

        chart = (
            alt.Chart(forecast_df)
            .mark_line(
                point=True,
                strokeWidth=3
            )
            .encode(
                x=alt.X(
                    "Month:O",
                    title="Month",
                    sort=list(range(1, 13)),
                    axis=alt.Axis(
                        labelAngle=0,
                        labelFontSize=12,
                        labelColor="#475569",
                        titleFontSize=13,
                        titleColor="#64748B",
                        domain=False,
                        tickSize=0
                    )
                ),
                y=alt.Y(
                    "Pain Score:Q",
                    title="Pain / difficulty score",
                    scale=alt.Scale(domain=[0, 5]),
                    axis=alt.Axis(
                        values=[0, 1, 2, 3, 4, 5],
                        labelFontSize=12,
                        labelColor="#64748B",
                        titleFontSize=13,
                        titleColor="#64748B",
                        grid=True,
                        gridColor="#E2E8F0",
                        domain=False,
                        tickSize=0
                    )
                ),
                tooltip=[
                    alt.Tooltip("Month:O", title="Month"),
                    alt.Tooltip("Pain Score:Q", title="Pain Score", format=".0f")
                ]
            )
            .properties(
                height=300,
                padding={"left": 10, "right": 20, "top": 10, "bottom": 10}
            )
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelFont="Arial",
                titleFont="Arial"
            )
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown(
            """
            <div style="
                font-size: 12px;
                color: #64748B;
                line-height: 1.5;
                margin-top: 8px;
            ">
                This hardcoded forecast is included for demonstration only. It should not be interpreted
                as a validated clinical pain prediction.
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# Page-level LLM preparation
# --------------------------------------------------
def prepare_physical_insights_data(user_data):
    """
    Runs all LLM work required for the Physical Insights page before rendering charts.
    """

    matrix_df, matrix_source = get_matrix_scores(user_data)

    return {
        "matrix_df": matrix_df,
        "matrix_source": matrix_source,
    }


def prepare_psychological_insights_data(user_data):
    """
    Runs all LLM work required for the Psychological Insights page before rendering charts.
    """

    big5_scores, score_source = get_radar_big5_scores(user_data)

    return {
        "big5_scores": big5_scores,
        "score_source": score_source,
    }


# --------------------------------------------------
# Pages
# --------------------------------------------------
def show_dashboard():
    st.markdown("<div class='main-title'>Clinical Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Structured AMM-based patient and exercise visualization dashboard.</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.user_data:
        st.info("Enter and load a patient ID from the Patient Profile page first.")
        return

    user_data = st.session_state.user_data

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Patient ID", get_patient_id(user_data))

    with col2:
        metric_card("Age", get_age(user_data))

    with col3:
        metric_card("Perceived Difficulty", f"{get_main_perceived_difficulty(user_data)}/5")

    with col4:
        metric_card("Actual Difficulty", f"{get_main_actual_difficulty(user_data)}/5")

    with st.spinner("Generating dashboard LLM scores with Llama 3.1 8B..."):
        psychological_data = prepare_psychological_insights_data(user_data)
        physical_data = prepare_physical_insights_data(user_data)

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        render_perceived_actual_chart(user_data)

    with right_col:
        render_personality_trait_radar(
            user_data,
            big5_scores=psychological_data["big5_scores"],
            score_source=psychological_data["score_source"]
        )

    
    render_exercise_suitability_matrix(
        user_data,
        matrix_df=physical_data["matrix_df"],
        score_source=physical_data["matrix_source"]
    )

    render_exercise_comparison_card(user_data)


def show_patient_profile():
    st.markdown("<div class='main-title'>Patient Profile</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Enter a patient ID to load structured data from the JSON dataset.</div>",
        unsafe_allow_html=True
    )

    # ID entry card
    # st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        patient_id = st.text_input(
            "Enter Patient ID",
            value=st.session_state.selected_patient_id or "AMM_0001",
            placeholder="Example: AMM_0001"
        )

    with col2:
        st.write("")
        st.write("")
        load_clicked = st.button("Load Patient", use_container_width=True)

    st.markdown("<div></div>", unsafe_allow_html=True)

    if load_clicked:
        user_data = load_patient_by_id(patient_id)

        if user_data is None:
            st.error(f"No patient found for ID: {patient_id}")
        else:
            st.session_state.selected_patient_id = patient_id
            st.session_state.user_data = user_data
            st.success(f"Loaded patient profile: {patient_id}")

    if not st.session_state.user_data:
        st.info("No patient loaded yet.")
        return

    user_data = st.session_state.user_data

    st.markdown("<div class='card-title'>Basic Patient Stats</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            **ID:** {get_patient_id(user_data)}  
            **Age:** {get_age(user_data)}  
            **Gender:** {get_gender(user_data)}  
            """,
            unsafe_allow_html=False
        )

    with col2:
        st.markdown(
            f"""
            **Overall Health:** {get_overall_health(user_data)}  
            **Mobility:** {get_mobility(user_data)}  
            **Disability:** {get_disability(user_data)}  
            """,
            unsafe_allow_html=False
        )

    with col3:
        st.markdown(
            f"""
            **Activities:** {get_activities(user_data)}  
            **Exercise/week:** {get_exercise_days(user_data)}  
            **Emotion:** {get_emotional_state(user_data)}  
            """,
            unsafe_allow_html=False
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Placeholder raw JSON area
    with st.expander("Raw JSON record placeholder"):
        st.json(user_data)


def show_exercise_insights():
    st.markdown("<div class='main-title'>Physical Insights</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Doctor-facing structured visualizations for exercise difficulty and suitability.</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.user_data:
        st.info("Load a patient profile first.")
        return

    user_data = st.session_state.user_data

    with st.spinner("Generating physical insight scores with Llama 3.1 8B..."):
        physical_data = prepare_physical_insights_data(user_data)

    col1, col2 = st.columns(2)

    with col1:
        render_perceived_actual_chart(user_data)

    with col2:
        render_exercise_comparison_card(user_data)

    render_exercise_suitability_matrix(
        user_data,
        matrix_df=physical_data["matrix_df"],
        score_source=physical_data["matrix_source"]
    )


def show_amm_view():
    st.markdown("<div class='main-title'>Psychological Insights</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Structured AMM dimensions mapped to clinical/rehabilitation concepts.</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.user_data:
        st.info("Load a patient profile first.")
        return

    user_data = st.session_state.user_data

    with st.spinner("Generating psychological insight scores with Llama 3.1 8B..."):
        psychological_data = prepare_psychological_insights_data(user_data)

    col1, col2 = st.columns(2)

    with col1:
        render_icf_bps_domain_profile(
            user_data,
            big5_scores=psychological_data["big5_scores"],
            score_source=psychological_data["score_source"]
        )

    with col2:
        render_personality_trait_radar(
            user_data,
            big5_scores=psychological_data["big5_scores"],
            score_source=psychological_data["score_source"]
        )


def show_reports():
    st.markdown("<div class='main-title'>Reports</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Placeholder page for exporting or reviewing structured dashboard summaries.</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.user_data:
        st.info("Load a patient profile first.")
        return

    chart_placeholder(
        "Report Preview",
        "Placeholder for structured doctor-facing report<br>No generated free text",
        height="small"
    )



# --------------------------------------------------
# Main app
# --------------------------------------------------
def main():
    inject_css()

    if not st.session_state.llama_loaded:
        initialize_llama_on_startup()

    page = render_sidebar()

    if page == "Dashboard":
        show_dashboard()

    elif page == "Patient Profile":
        show_patient_profile()

    elif page == "Physical Insights":
        show_exercise_insights()

    elif page == "Psychological Insights":
        show_amm_view()


if __name__ == "__main__":
    main()
