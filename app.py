"""
app.py - MediGuide AI Streamlit Application
=============================================
Main entry point for the MediGuide AI medical symptom assessment prototype.

Run with:
    streamlit run app.py

This file handles:
  - Streamlit page configuration and layout
  - Sidebar (app info, disclaimer, model config, language, cache settings)
  - Main input form (patient data collection)
  - Input validation
  - Calling the LLMChain for structured JSON assessment
  - Streaming narrative guidance via st.write_stream
  - Rendering the results dashboard with safety warnings

IMPORTANT: This is an EDUCATIONAL AI prototype, NOT a medical device.
It does NOT provide confirmed diagnoses.
"""

import streamlit as st
import time

# ---------------------------------------------------------------------------
# Import backend modules from src/
# ---------------------------------------------------------------------------
from src.config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    SYMPTOM_OPTIONS,
    DURATION_OPTIONS,
    GENDER_OPTIONS,
    LANGUAGE_OPTIONS,
    MEDICAL_DISCLAIMER,
    EMERGENCY_WARNING,
    URGENCY_LEVELS,
)
from src.chains import (
    get_llm,
    get_assessment_chain,
    run_assessment_chain,
    stream_narrative,
    demonstrate_messages,
)
from src.prompts import JSON_SCHEMA
from src.cache_manager import setup_cache
from src.utils import (
    safe_parse_json,
    validate_assessment,
    format_symptoms,
    validate_age,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MediGuide AI - Medical Symptom Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===========================================================================
# SIDEBAR
# ===========================================================================
def render_sidebar() -> dict:
    """
    Render the sidebar with app info, disclaimer, model config,
    language selection, and cache settings.

    Returns:
        A dictionary with sidebar configuration values.
    """
    with st.sidebar:
        # App name and description
        st.title("🏥 MediGuide AI")
        st.caption("AI-Powered Medical Symptom Assessment & Patient Guidance")
        st.markdown("---")

        # ----- Medical Disclaimer (prominent, in sidebar) -----
        st.warning(MEDICAL_DISCLAIMER)
        st.markdown("---")

        # ----- Runtime API Key -----
        st.subheader("🔐 OpenAI API Key")
        st.caption(
            "Enter your own OpenAI API key to use this app. "
            "The key is kept only in this Streamlit session and is not saved by the project."
        )
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            key="runtime_openai_api_key",
            help="Your key is required to make OpenAI API requests. Never share it or commit it to GitHub.",
        ).strip()

        if api_key:
            st.success("✅ API key entered for this session.")
        else:
            st.warning("🔑 Enter an OpenAI API key to enable the AI assessment.")

        def clear_api_key() -> None:
            st.session_state["runtime_openai_api_key"] = ""

        st.button(
            "🗑️ Clear API Key",
            use_container_width=True,
            on_click=clear_api_key,
        )

        st.markdown("---")

        # ----- Model Configuration -----
        st.subheader("⚙️ Model Configuration")

        selected_model = st.selectbox(
            "Select AI Model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
            help="Choose the OpenAI model to use for the assessment.",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_TEMPERATURE,
            step=0.1,
            help="Lower = more focused and deterministic. Higher = more creative.",
        )

        st.markdown("---")

        # ----- Language Selection -----
        st.subheader("🌐 Language")
        sidebar_language = st.selectbox(
            "Response Language",
            options=LANGUAGE_OPTIONS,
            index=0,
            help="The AI will respond in this language.",
            key="sidebar_language",
        )

        st.markdown("---")

        # ----- Cache Settings -----
        st.subheader("💾 Caching")
        cache_type = st.selectbox(
            "Cache Type",
            options=["None", "InMemoryCache", "SQLiteCache"],
            index=0,
            help=(
                "InMemoryCache: fast, stored in RAM, lost on restart.\n"
                "SQLiteCache: stored on disk, survives restart."
            ),
        )

        # Apply cache configuration
        cache_status = setup_cache(cache_type)
        with st.expander("Cache Status", expanded=False):
            st.markdown(cache_status)

        st.markdown("---")
        st.caption("© 2024 MediGuide AI — Educational Prototype")

    return {
        "model": selected_model,
        "temperature": temperature,
        "sidebar_language": sidebar_language,
        "cache_type": cache_type,
        "api_key": api_key,
    }


# ===========================================================================
# MAIN INPUT FORM
# ===========================================================================
def render_input_form(sidebar_language: str) -> dict | None:
    """
    Render the main patient input form.

    Uses all required Streamlit widgets:
      - text_input, selectbox, multiselect, slider, text_area, button

    Args:
        sidebar_language: Default language from the sidebar.

    Returns:
        A dictionary of form inputs if the form is submitted, else None.
    """
    st.title("🏥 MediGuide AI")
    st.subheader("AI-Powered Medical Symptom Assessment")

    # ----- Main Area Disclaimer -----
    st.info(MEDICAL_DISCLAIMER)

    st.markdown("---")
    st.subheader("📋 Patient Information Form")
    st.markdown(
        "Please fill in the details below. "
        "Fields marked with **\\*** are required."
    )

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)

    with col1:
        # Patient Age (text_input — assignment requirement)
        age_input = st.text_input(
            "Patient Age *",
            placeholder="e.g. 25",
            help="Enter the patient's age as a whole number.",
        )

        # Gender (selectbox — assignment requirement)
        gender = st.selectbox(
            "Gender *",
            options=GENDER_OPTIONS,
            help="Select the patient's gender.",
        )

        # Symptoms (multiselect — assignment requirement)
        selected_symptoms = st.multiselect(
            "Symptoms * (select all that apply)",
            options=SYMPTOM_OPTIONS,
            help="Choose one or more symptoms from the list.",
        )

        # Optional free-text symptoms
        free_text_symptoms = st.text_input(
            "Additional Symptoms (optional)",
            placeholder="e.g. ear pain, swollen glands",
            help="Type any symptoms not in the list above, separated by commas.",
        )

    with col2:
        # Duration (selectbox — assignment requirement)
        duration = st.selectbox(
            "Duration of Symptoms *",
            options=DURATION_OPTIONS,
            help="How long have the symptoms been present?",
        )

        # Severity (slider 1-10 — assignment requirement)
        severity = st.slider(
            "Severity (1 = mild, 10 = severe) *",
            min_value=1,
            max_value=10,
            value=5,
            help="Rate the overall severity of the symptoms.",
        )

        # Answer Language (selectbox — assignment requirement)
        language = st.selectbox(
            "Answer Language *",
            options=LANGUAGE_OPTIONS,
            index=LANGUAGE_OPTIONS.index(sidebar_language),
            help="The AI will respond in this language.",
            key="form_language",
        )

    # Full-width fields
    medical_conditions = st.text_area(
        "Existing Medical Conditions",
        placeholder="e.g. diabetes, hypertension, asthma",
        help="List any pre-existing medical conditions.",
    )

    medications = st.text_area(
        "Current Medications",
        placeholder="e.g. metformin 500mg, lisinopril 10mg",
        help="List any medications currently being taken.",
    )

    additional_notes = st.text_area(
        "Additional Notes",
        placeholder="e.g. recently travelled, family history of heart disease",
        help="Any other relevant information.",
    )

    st.markdown("---")

    # Submit button (button — assignment requirement)
    submitted = st.button(
        "🔍 Analyse Symptoms",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        return {
            "age_input": age_input,
            "gender": gender,
            "selected_symptoms": selected_symptoms,
            "free_text_symptoms": free_text_symptoms,
            "duration": duration,
            "severity": severity,
            "language": language,
            "medical_conditions": medical_conditions or "None reported",
            "medications": medications or "None reported",
            "additional_notes": additional_notes or "None",
        }

    return None


# ===========================================================================
# INPUT VALIDATION
# ===========================================================================
def validate_inputs(form_data: dict) -> tuple[bool, list[str]]:
    """
    Validate form inputs before calling the API.

    Args:
        form_data: The dictionary returned by render_input_form().

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    # Validate age
    age_valid, age_msg = validate_age(form_data["age_input"])
    if not age_valid:
        errors.append(age_msg)

    # Validate symptoms (must not be empty — assignment requirement)
    symptoms_combined = format_symptoms(
        form_data["selected_symptoms"],
        form_data["free_text_symptoms"],
    )
    if symptoms_combined == "None reported":
        errors.append(
            "Please select at least one symptom or type additional symptoms."
        )

    return len(errors) == 0, errors


# ===========================================================================
# RESULTS DASHBOARD
# ===========================================================================
def render_urgency_display(urgency: str) -> None:
    """
    Display the urgency level with appropriate colour and styling.

    Uses st.error, st.warning, st.info, st.success based on the level.
    """
    urgency = urgency.upper()

    if urgency == "EMERGENCY":
        st.error(EMERGENCY_WARNING)
    elif urgency == "HIGH":
        st.error(
            "🔴 **Urgency Level: HIGH**\n\n"
            "The symptoms described are concerning. "
            "Please seek medical attention **promptly**. "
            "This is an AI-generated preliminary assessment for "
            "educational purposes only — NOT a confirmed diagnosis."
        )
    elif urgency == "MEDIUM":
        st.warning(
            "🟡 **Urgency Level: MEDIUM**\n\n"
            "The symptoms described warrant professional medical evaluation "
            "within a reasonable timeframe. Please schedule an appointment "
            "with a healthcare professional. "
            "This is an AI-generated preliminary assessment for "
            "educational purposes only — NOT a confirmed diagnosis."
        )
    else:  # LOW
        st.success(
            "🟢 **Urgency Level: LOW**\n\n"
            "Based on this preliminary assessment, the symptoms described "
            "appear minor. Self-care and monitoring may be appropriate, "
            "but consulting a doctor is **still recommended**. "
            "This is an AI-generated preliminary assessment for "
            "educational purposes only — NOT a confirmed diagnosis."
        )


def render_results_dashboard(assessment: dict, form_data: dict) -> None:
    """
    Render the full results dashboard.

    Displays: patient summary, AI information, possible conditions,
    urgency, next steps, doctor questions, warning signs.

    Uses: st.metric, st.warning, st.info, st.error, st.success,
          st.expander, tabs, columns.

    Args:
        assessment: The validated JSON assessment dictionary.
        form_data: The original form inputs.
    """
    st.markdown("---")
    st.header("📊 Assessment Results")

    # ----- Results Disclaimer -----
    st.warning(
        "⚠️ **Reminder:** The following results are generated by an AI for "
        "**educational purposes only**. They are NOT a confirmed medical "
        "diagnosis. Always consult a qualified healthcare professional."
    )

    # ----- Urgency Level (visually prominent) -----
    render_urgency_display(assessment["urgency_level"])

    # ----- Tabs for organised results -----
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary",
        "🔬 Possible Conditions",
        "📝 Next Steps & Questions",
        "⚠️ Warning Signs",
    ])

    # ----- Tab 1: Patient Summary + AI Summary -----
    with tab1:
        st.subheader("Patient Symptom Summary")

        # Use columns and metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Age", form_data["age_input"])
        with col2:
            st.metric("Gender", form_data["gender"])
        with col3:
            st.metric("Severity", f"{form_data['severity']}/10")
        with col4:
            st.metric("Urgency", assessment["urgency_level"])

        symptoms_str = format_symptoms(
            form_data["selected_symptoms"],
            form_data["free_text_symptoms"],
        )
        st.markdown(f"**Symptoms:** {symptoms_str}")
        st.markdown(f"**Duration:** {form_data['duration']}")
        st.markdown(f"**Existing Conditions:** {form_data['medical_conditions']}")
        st.markdown(f"**Current Medications:** {form_data['medications']}")
        if form_data["additional_notes"] != "None":
            st.markdown(f"**Additional Notes:** {form_data['additional_notes']}")

        st.markdown("---")
        st.subheader("AI-Generated General Information")
        st.info(
            "ℹ️ The following is AI-generated educational information, "
            "**NOT** a medical diagnosis."
        )
        st.markdown(assessment["summary"])

    # ----- Tab 2: Possible Conditions (educational only) -----
    with tab2:
        st.subheader("Possible Conditions (Educational Information Only)")
        st.warning(
            "⚠️ These are **possible** conditions for **educational purposes only**. "
            "They are **NOT confirmed diagnoses**. Only a qualified healthcare "
            "professional can provide a diagnosis after proper examination."
        )

        if assessment["possible_conditions"]:
            for i, condition in enumerate(assessment["possible_conditions"], 1):
                with st.expander(
                    f"📌 {condition.get('name', 'Unknown')}",
                    expanded=(i == 1),
                ):
                    st.markdown(f"**Reason:** {condition.get('reason', 'N/A')}")
                    st.caption(
                        "This is educational information only — "
                        "not a confirmed diagnosis."
                    )
        else:
            st.info("No specific conditions were identified by the model.")

    # ----- Tab 3: Next Steps & Doctor Questions -----
    with tab3:
        col_steps, col_questions = st.columns(2)

        with col_steps:
            st.subheader("📝 Recommended Next Steps")
            if assessment["recommended_next_steps"]:
                for step in assessment["recommended_next_steps"]:
                    st.markdown(f"- {step}")
            else:
                st.info("No specific next steps were provided.")

        with col_questions:
            st.subheader("❓ Questions for Your Doctor")
            st.info(
                "Consider asking these questions at your next appointment:"
            )
            if assessment["questions_for_doctor"]:
                for q in assessment["questions_for_doctor"]:
                    st.markdown(f"- {q}")
            else:
                st.info("No specific questions were suggested.")

    # ----- Tab 4: Warning Signs -----
    with tab4:
        st.subheader("⚠️ Warning Signs Requiring Immediate Attention")

        if assessment["warning_signs"]:
            for sign in assessment["warning_signs"]:
                st.error(f"🚨 {sign}")
            st.markdown("---")
            st.error(
                "If you experience any of the above warning signs, "
                "**seek medical help immediately**. Do not wait."
            )
        else:
            st.info(
                "No specific warning signs were identified. However, "
                "if your condition worsens, please seek medical attention."
            )

    # ----- Final Disclaimer in Results -----
    st.markdown("---")
    st.warning(
        "⚠️ **FINAL REMINDER:** This assessment is generated by an AI system "
        "for **educational purposes only**. It is **NOT** a substitute for "
        "professional medical advice, diagnosis, or treatment. Always seek "
        "the advice of a qualified healthcare professional with any questions "
        "regarding a medical condition. If you think you may have a medical "
        "emergency, call your local emergency number immediately."
    )


# ===========================================================================
# MAIN APPLICATION
# ===========================================================================
def main():
    """
    Main application entry point.
    Orchestrates the sidebar, form, validation, API calls, and dashboard.
    """
    # Render sidebar and get configuration
    sidebar_config = render_sidebar()

    # Render the input form
    form_data = render_input_form(sidebar_config["sidebar_language"])

    # If form was not submitted, stop here
    if form_data is None:
        return

    # ----- Input Validation -----
    is_valid, errors = validate_inputs(form_data)
    if not is_valid:
        for err in errors:
            st.warning(f"⚠️ {err}")
        st.info("Please correct the above issues and try again.")
        return  # Do NOT call the API

    # ----- Check Runtime API Key -----
    api_key = sidebar_config["api_key"]
    if not api_key:
        st.error(
            "🔑 **OpenAI API key required.**\n\n"
            "Enter your own API key in the sidebar before analysing symptoms. "
            "The project does not contain or load a built-in API key."
        )
        return

    # ----- Prepare inputs for the chain -----
    symptoms_str = format_symptoms(
        form_data["selected_symptoms"],
        form_data["free_text_symptoms"],
    )

    chain_inputs = {
        "age": form_data["age_input"],
        "gender": form_data["gender"],
        "symptoms": symptoms_str,
        "duration": form_data["duration"],
        "severity": str(form_data["severity"]),
        "medical_conditions": form_data["medical_conditions"],
        "medications": form_data["medications"],
        "additional_notes": form_data["additional_notes"],
        "language": form_data["language"],
    }

    # ----- Create the LLM -----
    try:
        llm = get_llm(
            api_key=api_key,
            model=sidebar_config["model"],
            temperature=sidebar_config["temperature"],
        )
    except Exception as e:
        st.error(f"❌ Failed to initialize the AI model: {e}")
        return

    # ----- Run the structured assessment chain -----
    st.markdown("---")
    st.subheader("⏳ Generating Assessment...")

    try:
        start_time = time.time()
        chain = get_assessment_chain(llm)
        raw_response = run_assessment_chain(chain, chain_inputs)
        elapsed = time.time() - start_time
        st.caption(f"✅ Structured assessment generated in {elapsed:.1f}s")
    except Exception as e:
        st.error(
            f"❌ **API Error:** {e}\n\n"
            "Please check your API key and internet connection."
        )
        return

    # ----- Parse JSON safely -----
    parsed, parse_error = safe_parse_json(raw_response)

    if parse_error:
        st.error("❌ **Failed to parse the AI response.**")
        st.warning(parse_error)
        st.info(
            "The model's response could not be interpreted as valid JSON. "
            "Please try again. The raw output is shown above for debugging."
        )
        return

    # ----- Validate the parsed assessment -----
    assessment, validation_warnings = validate_assessment(parsed)

    if validation_warnings:
        with st.expander("⚠️ Response Validation Warnings", expanded=False):
            for w in validation_warnings:
                st.warning(w)

    # ----- Stream the narrative guidance -----
    st.markdown("---")
    st.subheader("💬 AI Health Guidance (Streaming)")
    st.info(
        "ℹ️ The following guidance is streamed live from the AI. "
        "This is **educational information only** — NOT a medical diagnosis."
    )

    try:
        st.write_stream(stream_narrative(llm, chain_inputs))
    except Exception as e:
        st.warning(
            f"⚠️ Streaming encountered an issue: {e}\n\n"
            "The structured assessment below is still available."
        )

    # ----- Render the results dashboard -----
    render_results_dashboard(assessment, form_data)

    # ----- Message Types Demonstration (inside an expander) -----
    with st.expander("🔬 LangChain Message Types Demo", expanded=False):
        st.markdown(
            "This section demonstrates `SystemMessage`, `HumanMessage`, "
            "and `AIMessage` — the three core message types in LangChain."
        )
        try:
            demo_result = demonstrate_messages(llm, symptoms_str)
            st.markdown(demo_result)
        except Exception as e:
            st.warning(f"Demo encountered an issue: {e}")

        st.caption(
            "This demonstrates how SystemMessage (role/rules), "
            "HumanMessage (user input), and AIMessage (AI response context) "
            "work together in a conversation."
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
