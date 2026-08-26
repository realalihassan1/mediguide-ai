"""
config.py - Application Configuration
======================================
Defines form options and project-wide constants.

API keys are intentionally NOT loaded from a file or hard-coded.
The Streamlit app asks the user to enter an OpenAI API key at runtime.
"""

# ---------------------------------------------------------------------------
# Default model settings
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.3
AVAILABLE_MODELS: list[str] = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]

# ---------------------------------------------------------------------------
# Symptom options for the multiselect widget
# ---------------------------------------------------------------------------
SYMPTOM_OPTIONS: list[str] = [
    "Headache", "Fever", "Cough", "Sore Throat", "Runny Nose",
    "Body Aches", "Fatigue", "Nausea", "Vomiting", "Diarrhea",
    "Shortness of Breath", "Chest Pain", "Dizziness", "Rash", "Joint Pain",
    "Back Pain", "Abdominal Pain", "Loss of Appetite", "Difficulty Sleeping",
    "Anxiety", "Blurred Vision", "Numbness or Tingling", "Swelling",
    "Weight Loss (unexplained)", "Palpitations",
]

# ---------------------------------------------------------------------------
# Duration options
# ---------------------------------------------------------------------------
DURATION_OPTIONS: list[str] = [
    "Less than 1 day",
    "1-3 days",
    "4-7 days",
    "1-2 weeks",
    "2-4 weeks",
    "1-3 months",
    "More than 3 months",
]

# ---------------------------------------------------------------------------
# Gender options
# ---------------------------------------------------------------------------
GENDER_OPTIONS: list[str] = ["Male", "Female", "Other", "Prefer not to say"]

# ---------------------------------------------------------------------------
# Language options
# ---------------------------------------------------------------------------
LANGUAGE_OPTIONS: list[str] = [
    "English", "Urdu", "Arabic", "Spanish", "French", "German",
]

# ---------------------------------------------------------------------------
# Safety / medical disclaimers
# ---------------------------------------------------------------------------
MEDICAL_DISCLAIMER: str = (
    "⚠️ This is an educational AI prototype, NOT a medical device. "
    "It does not provide confirmed diagnoses or replace professional medical advice. "
    "Always consult a qualified healthcare professional."
)

EMERGENCY_WARNING: str = (
    "🚨 **EMERGENCY WARNING:** If you are experiencing severe or life-threatening "
    "symptoms, seek emergency medical care immediately. Do not rely on this AI tool."
)

URGENCY_LEVELS: list[str] = ["LOW", "MEDIUM", "HIGH", "EMERGENCY"]
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SQLITE_CACHE_DB = BASE_DIR / "cache.db"

REQUIRED_JSON_KEYS: list[str] = [
    "assessment",
    "possible_conditions",
    "recommendations",
    "urgency_level",
]