"""
prompts.py - Prompt Templates
==============================
Contains:
  1. SYSTEM_PROMPT_TEXT  - safety rules & role definition (plain string)
  2. ASSESSMENT_PROMPT_TEMPLATE - a PromptTemplate (single-string with variables)
  3. ASSESSMENT_CHAT_TEMPLATE  - a ChatPromptTemplate (System + Human messages)
  4. NARRATIVE_CHAT_TEMPLATE   - a ChatPromptTemplate for streaming narrative
  5. JSON_SCHEMA               - the exact JSON structure the model must return

These are *reusable* templates — they are defined once and invoked with
different patient inputs every time the form is submitted.

IMPORTANT: This is an educational AI prototype, NOT a medical device.
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# ---------------------------------------------------------------------------
# JSON schema the model MUST return (Section 10 of the assignment)
# ---------------------------------------------------------------------------
JSON_SCHEMA: str = """{
  "summary": "<brief overview of the patient's situation>",
  "possible_conditions": [
    {
      "name": "<condition name>",
      "reason": "<why this condition might be relevant>"
    }
  ],
  "urgency_level": "<LOW | MEDIUM | HIGH | EMERGENCY>",
  "recommended_next_steps": ["<step 1>", "<step 2>"],
  "questions_for_doctor": ["<question 1>", "<question 2>"],
  "warning_signs": ["<sign 1>", "<sign 2>"]
}"""

# ---------------------------------------------------------------------------
# System prompt text (safety rules + role definition)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEXT: str = """You are MediGuide AI, an EDUCATIONAL medical information assistant.

=== CRITICAL SAFETY RULES (you MUST follow these at ALL times) ===
1. You are NOT a doctor. You are NOT a medical professional.
2. You must NEVER provide a confirmed diagnosis.
3. All "possible conditions" you mention are for EDUCATIONAL INFORMATION ONLY.
4. You must ALWAYS recommend that the user consult a qualified healthcare professional.
5. If symptoms indicate a possible EMERGENCY (e.g. chest pain, difficulty breathing, signs of stroke, severe bleeding), you MUST:
   - Set urgency_level to "EMERGENCY"
   - Clearly instruct the user to seek emergency medical help IMMEDIATELY
   - Do NOT downplay the situation
6. Do NOT provide false certainty. Use language like "may", "could", "possible", "might".
7. Highlight warning signs that require immediate medical attention.
8. Always remind the user that this is an AI-generated educational assessment, not a medical diagnosis.

=== RESPONSE LANGUAGE ===
You MUST respond in: {language}

=== OUTPUT FORMAT ===
Return ONLY valid JSON matching this exact schema (no markdown fences, no extra text):
{json_schema}

URGENCY LEVEL must be exactly one of: LOW, MEDIUM, HIGH, EMERGENCY
- LOW: Symptoms appear minor; self-care and monitoring may be appropriate, but consulting a doctor is still recommended.
- MEDIUM: Symptoms warrant professional medical evaluation within a reasonable timeframe.
- HIGH: Symptoms are concerning and medical attention should be sought promptly.
- EMERGENCY: Symptoms may indicate a life-threatening condition; immediate emergency help is required.
"""

# ---------------------------------------------------------------------------
# PromptTemplate (single-string template with variables)
#
# This satisfies the assignment requirement:
#   "A reusable single-string template with variables (age, gender, symptoms, ...)"
# ---------------------------------------------------------------------------
ASSESSMENT_PROMPT_TEMPLATE: PromptTemplate = PromptTemplate(
    input_variables=[
        "age",
        "gender",
        "symptoms",
        "duration",
        "severity",
        "medical_conditions",
        "medications",
        "additional_notes",
        "language",
    ],
    template=(
        "Patient Information:\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Symptoms: {symptoms}\n"
        "- Duration: {duration}\n"
        "- Severity (1-10): {severity}\n"
        "- Existing Medical Conditions: {medical_conditions}\n"
        "- Current Medications: {medications}\n"
        "- Additional Notes: {additional_notes}\n\n"
        "Please analyse the above patient information and provide a structured "
        "educational assessment in {language}. Remember: you are NOT diagnosing — "
        "you are providing general educational health information only.\n\n"
        "Return ONLY valid JSON matching the required schema."
    ),
)

# ---------------------------------------------------------------------------
# ChatPromptTemplate (System + Human messages)
#
# This satisfies the assignment requirement:
#   "A System + Human conversation carrying the safety rules and patient data."
# ---------------------------------------------------------------------------
ASSESSMENT_CHAT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        # SystemMessage: defines the AI's role + safety rules
        (
            "system",
            SYSTEM_PROMPT_TEXT,
        ),
        # HumanMessage: carries the patient data
        (
            "human",
            "Patient Information:\n"
            "- Age: {age}\n"
            "- Gender: {gender}\n"
            "- Symptoms: {symptoms}\n"
            "- Duration: {duration}\n"
            "- Severity (1-10): {severity}\n"
            "- Existing Medical Conditions: {medical_conditions}\n"
            "- Current Medications: {medications}\n"
            "- Additional Notes: {additional_notes}\n\n"
            "Please analyse this information and return ONLY valid JSON "
            "matching the required schema. Respond in {language}.",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Narrative ChatPromptTemplate (for streaming human-readable guidance)
#
# This template produces a natural-language narrative (not JSON).
# It is streamed live to the UI via st.write_stream().
# ---------------------------------------------------------------------------
NARRATIVE_CHAT_TEMPLATE: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are MediGuide AI, an EDUCATIONAL medical information assistant.\n\n"
            "CRITICAL RULES:\n"
            "- You are NOT a doctor. You do NOT provide confirmed diagnoses.\n"
            "- All information is for EDUCATIONAL purposes only.\n"
            "- Always recommend consulting a qualified healthcare professional.\n"
            "- If symptoms may indicate an emergency, clearly tell the user to "
            "seek emergency help IMMEDIATELY.\n"
            "- Use empathetic, clear, and calm language.\n"
            "- Respond in: {language}\n\n"
            "Write a helpful, human-readable narrative (NOT JSON). "
            "Use paragraphs and bullet points for readability. "
            "Include a reminder that this is educational information only.",
        ),
        (
            "human",
            "A patient has provided the following information:\n"
            "- Age: {age}\n"
            "- Gender: {gender}\n"
            "- Symptoms: {symptoms}\n"
            "- Duration: {duration}\n"
            "- Severity (1-10): {severity}\n"
            "- Existing Medical Conditions: {medical_conditions}\n"
            "- Current Medications: {medications}\n"
            "- Additional Notes: {additional_notes}\n\n"
            "Please provide a clear, empathetic, educational health guidance "
            "narrative in {language}. Remember to include disclaimers.",
        ),
    ]
)
