Streamlit link: https://mediguide-ai-byah.streamlit.app/

# 🏥 MediGuide AI

## AI-Powered Medical Symptom Assessment and Patient Guidance Assistant

> **⚠️ IMPORTANT:** This is an **educational AI prototype** only. It is **NOT** a medical device, not a replacement for a licensed doctor, and must **never** be used for real diagnosis or treatment. Always consult a qualified healthcare professional.

---

## 📖 Project Overview

MediGuide AI is an intelligent Streamlit application built with LangChain and OpenAI. Users enter basic health information and symptoms, and the AI generates structured, safety-focused preliminary guidance for **educational purposes only**.

The project demonstrates key LangChain concepts including prompt templates, chains, structured output, streaming, and caching — all wrapped in a professional, safety-first user interface.

---

## ✨ Features

- **Patient Data Collection** — Age, gender, symptoms, duration, severity, conditions, medications
- **Structured JSON Assessment** — AI returns validated JSON with conditions, urgency, next steps
- **Live Streaming Guidance** — Natural typing effect using `.stream()` + `st.write_stream()`
- **Urgency Level Display** — Color-coded LOW / MEDIUM / HIGH / EMERGENCY indicators
- **Dual Caching** — InMemoryCache (RAM) and SQLiteCache (disk) for performance
- **Multi-Language Support** — English, Urdu, Arabic, Spanish, French, and more
- **Safety-First Design** — Medical disclaimers on every screen; never claims a diagnosis
- **Robust Error Handling** — Safe JSON parsing; graceful API error recovery
- **LangChain Concepts** — Demonstrates all required concepts (see below)

---

## 🏗️ Architecture

```
Data Flow:
  Form Input → Validate → Build Prompt → Select Cache → LLMChain → Parse JSON → Stream Narrative → Render Dashboard
```

### Folder Structure

```
medical_ai_assistant/
├── app.py                  # Streamlit UI — run this
├── requirements.txt        # Python dependencies
├── .env.example            # Explains that no .env file is required
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # Settings, form options, constants
│   ├── prompts.py          # PromptTemplate, ChatPromptTemplate, JSON schema
│   ├── chains.py           # ChatOpenAI, LLMChain, streaming
│   ├── cache_manager.py    # InMemoryCache + SQLiteCache management
│   └── utils.py            # Safe JSON parsing, validation, helpers
└── docs/
    └── Medical_AI_Assignment.pdf
```

---

## 🚀 Installation & Setup

### 1. Prerequisites

- **Python 3.10+** installed
- **pip** package manager
- An **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd medical_ai_assistant
```

### 3. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Enter Your API Key at Runtime

This project intentionally does **not** contain an OpenAI API key and does not require a `.env` file.

When you run the app, enter **your own OpenAI API key** in the **🔐 OpenAI API Key** field in the Streamlit sidebar. The key is kept in the current Streamlit session and is not written into the project files.

> **⚠️ NEVER** paste your API key into Python source code, README files, GitHub, screenshots, or other public places.

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🔗 LangChain Concepts Demonstrated

### 1. ChatOpenAI

**What it is:** A wrapper around OpenAI's chat models (e.g., `gpt-4o-mini`).

**Where:** `src/chains.py` → `get_llm()`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0.3)
```

The `ChatOpenAI` object is used throughout the app — in the LLMChain, for streaming, and for the message types demo.

---

### 2. PromptTemplate

**What it is:** A reusable single-string template with placeholder variables. You define the template once and fill in different values each time.

**Where:** `src/prompts.py` → `ASSESSMENT_PROMPT_TEMPLATE`

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["age", "gender", "symptoms", ...],
    template="Patient Age: {age}\nGender: {gender}\nSymptoms: {symptoms}..."
)
```

**Why it matters:** Instead of manually formatting strings every time, `PromptTemplate` provides a consistent, reusable way to build prompts. It validates that all required variables are provided.

---

### 3. ChatPromptTemplate

**What it is:** A template that creates a conversation with distinct roles (system, human, AI). It carries both safety rules (in the system message) and patient data (in the human message).

**Where:** `src/prompts.py` → `ASSESSMENT_CHAT_TEMPLATE` and `NARRATIVE_CHAT_TEMPLATE`

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are MediGuide AI... Safety rules... {language}"),
    ("human",  "Patient Age: {age}, Symptoms: {symptoms}...")
])
```

**Why it matters:** Chat models work best when messages are structured by role. The system message sets persistent rules, while the human message carries the specific request.

---

### 4. SystemMessage, HumanMessage, AIMessage

**What they are:** The three core message types in LangChain conversations.

| Message Type | Purpose | Example |
|---|---|---|
| `SystemMessage` | Defines the AI's role, rules, and constraints | "You are MediGuide AI. Never diagnose." |
| `HumanMessage` | Carries the user's input/question | "Patient has fever and cough..." |
| `AIMessage` | Represents a previous AI response (for context) | "I will now analyse the symptoms..." |

**Where:** `src/chains.py` → `demonstrate_messages()`

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are an educational medical assistant..."),
    HumanMessage(content="Patient summary: ..."),
    AIMessage(content="Thank you, I will now analyse..."),
    HumanMessage(content="What questions should I ask my doctor?"),
]
response = llm.invoke(messages)
```

**Why AIMessage matters:** It shows the model what it "previously said", maintaining conversation context. This is essential for multi-turn conversations.

---

### 5. LLMChain

**What it is:** A reusable pipeline that combines a prompt template with an LLM. You define it once and run it with different inputs.

**Where:** `src/chains.py` → `get_assessment_chain()`

```python
from langchain_classic.chains import LLMChain

chain = LLMChain(llm=llm, prompt=ASSESSMENT_CHAT_TEMPLATE)
result = chain.run(inputs)
```

**Why it matters:** `LLMChain` encapsulates the prompt + model into a single reusable unit. You can swap models, change prompts, or add preprocessing without modifying the calling code.

> **Note:** In the current LangChain ecosystem, `LLMChain` is available via `langchain-classic`. The concept remains the same — it's a chain that combines a prompt with an LLM.

---

### 6. Structured JSON Output

**What it is:** Instructing the model to return a specific JSON schema, then parsing and validating the response.

**Where:**
- Schema definition: `src/prompts.py` → `JSON_SCHEMA`
- Safe parsing: `src/utils.py` → `safe_parse_json()`
- Validation: `src/utils.py` → `validate_assessment()`

```json
{
  "summary": "...",
  "possible_conditions": [{"name": "...", "reason": "..."}],
  "urgency_level": "LOW | MEDIUM | HIGH | EMERGENCY",
  "recommended_next_steps": ["..."],
  "questions_for_doctor": ["..."],
  "warning_signs": ["..."]
}
```

The system prompt instructs the model to return ONLY valid JSON. The parser handles edge cases:
- Strips Markdown ` ```json ` fences
- Extracts JSON from surrounding text
- Falls back gracefully on parse failure
- Validates all required keys and types

---

### 7. Streaming

**What it is:** Using the model's `.stream()` method to yield response chunks incrementally, creating a live "typing" effect.

**Where:** `src/chains.py` → `stream_narrative()`

```python
def stream_narrative(llm, inputs):
    messages = NARRATIVE_CHAT_TEMPLATE.format_messages(**inputs)
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content
```

Connected to Streamlit via:
```python
st.write_stream(stream_narrative(llm, chain_inputs))
```

**Why it matters:** Users see the response appearing word-by-word instead of waiting for the entire answer. This improves perceived responsiveness.

**Design note:** The streaming narrative uses a SEPARATE template (plain text) from the JSON assessment to avoid breaking JSON parsing.

---

### 8. Caching (InMemoryCache vs SQLiteCache)

**What it is:** Storing model responses so identical requests don't hit the API again.

**Where:** `src/cache_manager.py` → `setup_cache()`

| Feature | InMemoryCache | SQLiteCache |
|---|---|---|
| **Stored in** | RAM (memory) | A `.db` file on disk |
| **Speed** | Fastest | Fast, slightly slower |
| **Survives restart?** | ❌ No | ✅ Yes |
| **Best for** | One session / demos | Reusing across sessions |

```python
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

# Option 1: In-memory
set_llm_cache(InMemoryCache())

# Option 2: SQLite
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

**How it works:** `set_llm_cache()` registers the cache globally. LangChain then automatically checks the cache before every model call. If the same prompt was seen before, the cached result is returned instantly.

**Testing cache:** Submit the same form twice with caching enabled — the second run should be noticeably faster.

---

## 🛡️ Safety & Disclaimers

This application implements multiple layers of safety:

1. **Sidebar Disclaimer** — Always visible warning in the sidebar
2. **Main Area Disclaimer** — Prominent notice above the input form
3. **Results Disclaimer** — Warning before and after the results dashboard
4. **System Prompt Rules** — The AI is instructed to never diagnose
5. **Emergency Handling** — EMERGENCY urgency triggers immediate help instructions
6. **Educational Language** — Uses "possible", "may", "could" — never certainty
7. **Per-Condition Disclaimer** — Each possible condition is labelled as educational only

> **This is NOT a medical device. It is NOT a replacement for a licensed doctor.**

---

## 🧪 Testing Instructions

### Test Scenarios (from Assignment Section 18)

| # | Input | Expected Behaviour |
|---|---|---|
| 1 | Age 25, runny nose + sore throat, 1–3 days, severity 2 | Urgency **LOW**; calm monitoring advice |
| 2 | Age 40, fever + cough, 4–7 days, severity 6 | Urgency **MEDIUM/HIGH**; advises seeing a professional |
| 3 | Severe chest pain + shortness of breath | Urgency **HIGH/EMERGENCY**; urges immediate help |
| 4 | Submit same form twice (cache on) | Second run is faster; identical result |
| 5 | Empty symptoms | App warns the user and does NOT call the API |
| 6 | Language = Urdu | Guidance text returns in Urdu |

### How to Test

1. **Run the app:** `streamlit run app.py`
2. **Scenario 1:** Enter age 25, select "Runny Nose" + "Sore Throat", duration "1-3 days", severity 2. Verify LOW urgency.
3. **Scenario 2:** Enter age 40, select "Fever" + "Cough", duration "4-7 days", severity 6. Verify MEDIUM or HIGH urgency.
4. **Scenario 3:** Select "Chest Pain" + "Shortness of Breath", severity 9. Verify HIGH or EMERGENCY urgency.
5. **Scenario 4:** Enable InMemoryCache or SQLiteCache in the sidebar. Submit the same form twice. Verify faster second response.
6. **Scenario 5:** Leave all symptom fields empty and click submit. Verify warning message appears and no API call is made.
7. **Scenario 6:** Set language to "Urdu" and submit. Verify the response is in Urdu.

---

## 🔧 Troubleshooting

### "OpenAI API key required"
- Enter your own OpenAI API key in the **🔐 OpenAI API Key** field in the Streamlit sidebar.
- The key is not stored in the project files.
- If you clear it or restart the app, enter it again when needed.

### "ModuleNotFoundError"
- Make sure you activated your virtual environment
- Run: `pip install -r requirements.txt`

### "Could not parse JSON"
- This can happen occasionally with LLM responses
- Click "Analyse Symptoms" again — results vary per call
- The raw model output is shown for debugging

### "API Error / Rate Limit"
- Check your OpenAI account has credits
- Wait a moment and try again
- Consider using `gpt-3.5-turbo` for lower cost

### App won't start
- Ensure you're in the `medical_ai_assistant/` directory
- Run: `streamlit run app.py` (not `python app.py`)

---

## ✅ Assignment Requirements Checklist

| Requirement | Status | Location |
|---|---|---|
| Streamlit UI/UX (15 marks) | ✅ | `app.py` |
| ChatOpenAI integration (10 marks) | ✅ | `src/chains.py` |
| PromptTemplate (10 marks) | ✅ | `src/prompts.py` |
| ChatPromptTemplate + messages (10 marks) | ✅ | `src/prompts.py`, `src/chains.py` |
| LLMChain (10 marks) | ✅ | `src/chains.py` |
| Structured JSON output (10 marks) | ✅ | `src/prompts.py`, `src/utils.py` |
| Streaming (10 marks) | ✅ | `src/chains.py`, `app.py` |
| Caching (10 marks) | ✅ | `src/cache_manager.py` |
| Code quality & documentation (10 marks) | ✅ | All files |
| Testing & creativity (5 marks) | ✅ | README + test scenarios |

---

## 📜 Disclaimer

> **This project is for educational purposes only.**
> It is **NOT** a medical device and must **NOT** be used for real diagnosis or treatment.
> The AI-generated information is preliminary and for educational use only.
> Always consult a qualified healthcare professional for medical advice.
> In an emergency, call your local emergency number immediately.

---

## 📦 Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| LLM Framework | LangChain (langchain-classic, langchain-openai, langchain-community, langchain-core) |
| Model Provider | OpenAI (gpt-4o-mini) |
| User Interface | Streamlit |
| Secrets | python-dotenv |
| Caching | InMemoryCache + SQLiteCache |

---

*Built as a LangChain + Streamlit programming assignment — Course Module: Building LLM Applications with LangChain*
