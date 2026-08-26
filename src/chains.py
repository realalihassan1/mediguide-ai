"""
chains.py - LangChain Chains and Model Integration
====================================================
Contains:
  1. get_llm()                - Creates a ChatOpenAI instance.
  2. get_assessment_chain()   - Builds an LLMChain for structured JSON assessment.
  3. run_assessment_chain()   - Runs the assessment chain and returns raw text.
  4. stream_narrative()       - Streaming generator for the narrative guidance.
  5. demonstrate_messages()   - Demonstrates SystemMessage/HumanMessage/AIMessage.

LangChain concepts demonstrated:
  - ChatOpenAI: the OpenAI chat model wrapper
  - LLMChain: a reusable chain combining a prompt template and an LLM
  - PromptTemplate & ChatPromptTemplate: reusable prompt templates
  - SystemMessage, HumanMessage, AIMessage: role-based messages
  - .stream(): streaming chunks for live typing effect
  - Structured JSON output: forcing the model to return valid JSON

IMPORTANT: This is an educational AI prototype, NOT a medical device.
"""

from typing import Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# LLMChain lives in langchain-classic in this environment.
# The assignment requires "at least one reusable chain (LLMChain)".
# langchain-classic provides the original LLMChain API.
from langchain_classic.chains import LLMChain

from src.prompts import (
    ASSESSMENT_CHAT_TEMPLATE,
    NARRATIVE_CHAT_TEMPLATE,
    JSON_SCHEMA,
)


def get_llm(
    api_key: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> ChatOpenAI:
    """
    Create and return a ChatOpenAI instance.

    Args:
        api_key: The OpenAI API key entered by the user at runtime. It is never hard-coded or loaded from disk.
        model: The model name (e.g. "gpt-4o-mini").
        temperature: Controls randomness (lower = more deterministic).

    Returns:
        A configured ChatOpenAI object ready for use in chains.

    This satisfies the assignment requirement:
        "Integrate the OpenAI chat model (ChatOpenAI)."
    """
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


def get_assessment_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Build a reusable LLMChain for the structured JSON assessment.

    The chain combines:
      - ASSESSMENT_CHAT_TEMPLATE (ChatPromptTemplate with System + Human)
      - The ChatOpenAI model

    This satisfies the assignment requirement:
        "At least one reusable chain (LLMChain) for the assessment."

    Args:
        llm: A configured ChatOpenAI instance.

    Returns:
        An LLMChain that accepts patient input variables and returns
        a structured JSON assessment.
    """
    chain = LLMChain(
        llm=llm,
        prompt=ASSESSMENT_CHAT_TEMPLATE,
        verbose=False,
    )
    return chain


def run_assessment_chain(chain: LLMChain, inputs: dict) -> str:
    """
    Run the assessment chain with the given patient inputs.

    Args:
        chain: The LLMChain created by get_assessment_chain().
        inputs: A dictionary of patient data variables matching the
                template's input_variables.

    Returns:
        The raw text response from the model (expected to be JSON).

    Raises:
        Exception: If the API call fails (handled gracefully in app.py).
    """
    # Add the JSON schema to the inputs so the system prompt can reference it
    inputs_with_schema = {**inputs, "json_schema": JSON_SCHEMA}

    # Run the chain — LLMChain.run() returns the text output
    result = chain.run(inputs_with_schema)
    return result


def stream_narrative(
    llm: ChatOpenAI,
    inputs: dict,
) -> Generator[str, None, None]:
    """
    Stream human-readable narrative guidance using the model's .stream() method.

    This generator yields content chunks that can be fed directly into
    Streamlit's st.write_stream() for a natural typing effect.

    The narrative uses a SEPARATE template (NARRATIVE_CHAT_TEMPLATE) that
    instructs the model to return plain text (NOT JSON), so streaming
    does not break JSON parsing.

    Args:
        llm: A configured ChatOpenAI instance.
        inputs: Patient data variables.

    Yields:
        String chunks of the narrative as they arrive from the model.

    This satisfies the assignment requirement:
        "Stream the final guidance using .stream() into a Streamlit component."
    """
    # Format the narrative prompt messages
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)

    # Use .stream() to get chunks incrementally
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


def demonstrate_messages(llm: ChatOpenAI, patient_summary: str) -> str:
    """
    Demonstrate the use of SystemMessage, HumanMessage, and AIMessage.

    This function shows how the three message types work together
    in a conversation context:
      - SystemMessage: defines the AI's role and rules
      - HumanMessage: carries the user's input
      - AIMessage: represents a previous AI response (for context)

    This satisfies the assignment requirement:
        "Define the role + safety rules and send patient data;
         show how AIMessage fits a conversation."

    Args:
        llm: A configured ChatOpenAI instance.
        patient_summary: A summary of patient symptoms.

    Returns:
        The model's response as a string.
    """
    messages = [
        # SystemMessage: defines the role and safety rules
        SystemMessage(
            content=(
                "You are MediGuide AI, an educational medical information assistant. "
                "You are NOT a doctor. You must NEVER provide a confirmed diagnosis. "
                "All information is for educational purposes only. "
                "Always recommend consulting a qualified healthcare professional."
            )
        ),
        # HumanMessage: carries the patient's data
        HumanMessage(
            content=f"Here is a patient summary: {patient_summary}"
        ),
        # AIMessage: shows the AI's previous acknowledgement (context)
        # This demonstrates how AIMessage fits into the conversation flow.
        AIMessage(
            content=(
                "Thank you for providing this information. I will now analyse "
                "the symptoms and provide educational health guidance. "
                "Please remember that this is not a medical diagnosis."
            )
        ),
        # HumanMessage: follow-up request
        HumanMessage(
            content=(
                "Based on the symptoms above, what are the most important "
                "questions I should ask my doctor? Please provide a brief list."
            )
        ),
    ]

    # Invoke the model with the message list
    response = llm.invoke(messages)
    return response.content
