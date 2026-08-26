"""
utils.py - Utility Functions
==============================
Contains:
  1. safe_parse_json()  - robust JSON parsing with fence stripping
  2. validate_assessment() - validates the parsed JSON structure
  3. format_symptoms()  - combines multiselect + free-text symptoms
  4. validate_age()     - basic age input validation

These helpers ensure the app NEVER crashes due to malformed model output.

IMPORTANT: This is an educational AI prototype, NOT a medical device.
"""

import json
import re
from typing import Any

from src.config import REQUIRED_JSON_KEYS, URGENCY_LEVELS


def safe_parse_json(raw_text: str) -> tuple[dict | None, str | None]:
    """
    Safely parse a JSON string from the model's response.

    Handles common issues:
      - Markdown ```json ... ``` fences
      - Leading/trailing whitespace or non-JSON text
      - Completely invalid JSON

    Args:
        raw_text: The raw string returned by the LLM.

    Returns:
        A tuple of (parsed_dict, error_message).
        - On success: (dict, None)
        - On failure: (None, "friendly error message")

    The raw text is preserved for debugging when parsing fails.
    """
    if not raw_text or not raw_text.strip():
        return None, "The model returned an empty response."

    cleaned = raw_text.strip()

    # ------------------------------------------------------------------
    # Step 1: Remove Markdown JSON fences (```json ... ``` or ``` ... ```)
    # ------------------------------------------------------------------
    # Pattern matches ```json\n...\n``` or ```\n...\n```
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # ------------------------------------------------------------------
    # Step 2: Try to find the outermost JSON object { ... }
    # ------------------------------------------------------------------
    # If there's surrounding text, extract just the JSON object
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]

    # ------------------------------------------------------------------
    # Step 3: Attempt to parse the JSON
    # ------------------------------------------------------------------
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return None, (
                "The model returned valid JSON but it is not a dictionary. "
                f"Raw output preserved for debugging:\n\n```\n{raw_text}\n```"
            )
        return parsed, None
    except json.JSONDecodeError as e:
        return None, (
            f"Could not parse the model's response as JSON.\n\n"
            f"**Error:** {e}\n\n"
            f"**Raw model output (for debugging):**\n\n```\n{raw_text}\n```"
        )


def validate_assessment(data: dict) -> tuple[dict, list[str]]:
    """
    Validate the parsed assessment dictionary against the required schema.

    Checks:
      - All required keys are present
      - urgency_level is one of the allowed values
      - possible_conditions is a list of dicts with 'name' and 'reason'
      - Lists are actually lists

    Args:
        data: The parsed JSON dictionary.

    Returns:
        A tuple of (validated_data, list_of_warnings).
        The data is patched with defaults for missing fields so
        the app can still display partial results.
    """
    warnings: list[str] = []

    # Check required keys and fill defaults
    for key in REQUIRED_JSON_KEYS:
        if key not in data:
            warnings.append(f"Missing key in response: '{key}'")
            # Provide safe defaults
            if key == "summary":
                data[key] = "No summary was provided by the model."
            elif key == "possible_conditions":
                data[key] = []
            elif key == "urgency_level":
                data[key] = "MEDIUM"
            elif key in ("recommended_next_steps", "questions_for_doctor", "warning_signs"):
                data[key] = []

    # Validate urgency_level
    if data.get("urgency_level", "").upper() not in URGENCY_LEVELS:
        warnings.append(
            f"Invalid urgency_level: '{data.get('urgency_level')}'. "
            f"Expected one of {URGENCY_LEVELS}. Defaulting to MEDIUM."
        )
        data["urgency_level"] = "MEDIUM"
    else:
        # Normalise to uppercase
        data["urgency_level"] = data["urgency_level"].upper()

    # Validate possible_conditions is a list
    if not isinstance(data.get("possible_conditions"), list):
        warnings.append("'possible_conditions' should be a list. Resetting to empty.")
        data["possible_conditions"] = []
    else:
        # Validate each condition has 'name' and 'reason'
        for i, cond in enumerate(data["possible_conditions"]):
            if not isinstance(cond, dict):
                warnings.append(f"Condition #{i+1} is not a dict. Skipping.")
                continue
            if "name" not in cond:
                cond["name"] = "Unknown condition"
                warnings.append(f"Condition #{i+1} is missing 'name'.")
            if "reason" not in cond:
                cond["reason"] = "No reason provided."
                warnings.append(f"Condition #{i+1} is missing 'reason'.")

    # Validate lists
    for key in ("recommended_next_steps", "questions_for_doctor", "warning_signs"):
        if not isinstance(data.get(key), list):
            warnings.append(f"'{key}' should be a list. Resetting to empty.")
            data[key] = []

    return data, warnings


def format_symptoms(selected_symptoms: list[str], free_text: str) -> str:
    """
    Combine the multiselect symptoms and optional free-text symptoms
    into a single comma-separated string.

    Args:
        selected_symptoms: Symptoms chosen from the multiselect widget.
        free_text: Optional additional symptoms typed by the user.

    Returns:
        A combined symptoms string, or "None reported" if empty.
    """
    all_symptoms: list[str] = list(selected_symptoms)

    # Add free-text symptoms (split by comma)
    if free_text and free_text.strip():
        extra = [s.strip() for s in free_text.split(",") if s.strip()]
        all_symptoms.extend(extra)

    return ", ".join(all_symptoms) if all_symptoms else "None reported"


def validate_age(age_input: str) -> tuple[bool, str]:
    """
    Validate the age input from the text_input widget.

    Args:
        age_input: The raw string from st.text_input.

    Returns:
        A tuple of (is_valid, message).
        - On success: (True, "25")
        - On failure: (False, "error message")
    """
    if not age_input or not age_input.strip():
        return False, "Please enter the patient's age."

    cleaned = age_input.strip()

    try:
        age = int(cleaned)
    except ValueError:
        return False, "Age must be a whole number (e.g. 25)."

    if age < 0 or age > 150:
        return False, "Please enter a reasonable age (0–150)."

    return True, cleaned
