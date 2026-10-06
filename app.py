import os
import io
import re
import hashlib
import tempfile

import streamlit as st
import speech_recognition as sr
import pyttsx3
import pandas as pd

from pypdf import PdfReader
from langchain_community.llms import Ollama

from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import (
    store_embeddings,
    search,
    get_collection_count
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PersonaPrep AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {
        background: #f5f9ff;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #0b63f6 !important;
        min-width: 285px !important;
        max-width: 300px !important;
    }

    section[data-testid="stSidebar"] > div {
        background: #0b63f6 !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: #0b63f6 !important;
    }

    /* Sidebar normal text */

    section[data-testid="stSidebar"] p {
        color: white !important;
    }

    section[data-testid="stSidebar"] span {
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    section[data-testid="stSidebar"] small {
        color: rgba(255,255,255,0.80) !important;
    }


    /* ========================================================
       BRAND
       ======================================================== */

    .brand {
        text-align: center;
        padding: 8px 4px 20px 4px;
    }

    .brand-icon {
        font-size: 42px;
        line-height: 1;
        margin-bottom: 8px;
    }

    .brand-title {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff !important;
    }

    .brand-subtitle {
        font-size: 13px;
        color: rgba(255,255,255,0.85) !important;
        margin-top: 4px;
    }


    /* ========================================================
       SIDEBAR DIVIDER
       ======================================================== */

    .side-divider {
        height: 1px;
        background: rgba(255,255,255,0.25);
        margin: 15px 0;
    }


    /* ========================================================
       SIDEBAR HEADING
       ======================================================== */

    .side-heading {
        font-size: 11px;
        font-weight: 800;
        color: rgba(255,255,255,0.72) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 15px 0 7px 2px;
    }


    /* ========================================================
       SIDEBAR RADIO / MODES
       ======================================================== */

    section[data-testid="stSidebar"]
    div[role="radiogroup"] {
        gap: 4px !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label {
        background: transparent !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 9px 10px !important;
        margin: 2px 0 !important;
        cursor: pointer !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.15) !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label p {
        color: white !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"]
    div[role="radiogroup"] label span {
        color: white !important;
    }


    /* ========================================================
       SIDEBAR BUTTONS
       ======================================================== */

    section[data-testid="stSidebar"] button {
        background: white !important;
        color: #17365f !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] button p {
        color: #17365f !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background: #edf5ff !important;
        color: #0b63f6 !important;
    }


    /* ========================================================
       SIDEBAR SELECTBOX
       ======================================================== */

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] {
        background: white !important;
        border-radius: 12px !important;
        border: none !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {
        color: #17365f !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] input {
        color: #17365f !important;
    }

    div[data-baseweb="popover"] {
        z-index: 999999 !important;
    }

    div[data-baseweb="popover"] * {
        color: #17365f !important;
    }


    /* ========================================================
       SIDEBAR FILE UPLOADER
       ======================================================== */

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        padding: 5px !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] p {
        color: white !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stFileUploader"] span {
        color: white !important;
    }


    /* ========================================================
       SIDEBAR TOGGLE
       ======================================================== */

    section[data-testid="stSidebar"]
    [data-testid="stWidgetLabel"] {
        color: white !important;
    }

    section[data-testid="stSidebar"]
    [data-testid="stWidgetLabel"] p {
        color: white !important;
    }


    /* ========================================================
       DOCUMENT PILL
       ======================================================== */

    .document-pill {
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 8px 10px;
        margin: 5px 0;
        font-size: 12px;
        color: white !important;
    }


    /* ========================================================
       MAIN CONTAINER
       ======================================================== */

    .main-container {
        max-width: 1100px;
        margin: auto;
        padding: 8px 35px 30px 35px;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .top-title {
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        color: #15365f;
        margin-top: 4px;
    }

    .top-subtitle {
        text-align: center;
        color: #70839b;
        font-size: 14px;
        margin-bottom: 22px;
    }


    /* ========================================================
       WELCOME CARD
       ======================================================== */

    .welcome-card {
        background: linear-gradient(
            135deg,
            #0b63f6,
            #3985ff
        );
        border-radius: 26px;
        padding: 40px 30px;
        color: white;
        text-align: center;
        box-shadow: 0 12px 35px rgba(20,91,200,0.20);
        margin-bottom: 28px;
    }

    .welcome-card h1 {
        color: white !important;
        font-size: 30px;
        margin: 8px 0;
    }

    .welcome-card p {
        color: rgba(255,255,255,0.90) !important;
        font-size: 15px;
    }


    /* ========================================================
       CHAT BUBBLES
       ======================================================== */

    .user-label {
        font-weight: 700;
        color: #1769e8;
        margin-bottom: 5px;
        margin-top: 10px;
    }

    .assistant-label {
        font-weight: 700;
        color: #24364b;
        margin-bottom: 5px;
        margin-top: 10px;
    }

    .user-bubble {
        background: #0b63f6;
        color: white;
        padding: 13px 18px;
        border-radius: 20px 20px 5px 20px;
        margin: 5px 0 18px auto;
        max-width: 78%;
        width: fit-content;
        line-height: 1.5;
        box-shadow: 0 5px 15px rgba(11,99,246,0.15);
    }

    .assistant-bubble {
        background: white;
        color: #26384d;
        padding: 15px 18px;
        border-radius: 20px 20px 20px 5px;
        margin: 5px auto 20px 0;
        max-width: 82%;
        width: fit-content;
        line-height: 1.55;
        box-shadow: 0 5px 20px rgba(30,70,120,0.08);
        border: 1px solid #e6edf6;
    }

    .source-pill {
        display: inline-block;
        background: #eaf3ff;
        color: #1769e8;
        padding: 4px 9px;
        border-radius: 20px;
        font-size: 11px;
        margin-top: -10px;
        margin-bottom: 15px;
    }


    /* ========================================================
       MODE CARD
       ======================================================== */

    .mode-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #e3ebf5;
        box-shadow: 0 8px 25px rgba(40,80,130,0.07);
        margin-bottom: 20px;
    }

    .mode-title {
        color: #15365f !important;
        font-size: 22px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .mode-description {
        color: #71839a !important;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ========================================================
       VOICE SCREEN
       ======================================================== */

    .voice-screen {
        background: linear-gradient(
            145deg,
            #075ee9,
            #3987ff
        );
        border-radius: 30px;
        padding: 60px 30px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 40px rgba(0,90,220,0.20);
        margin-top: 20px;
        margin-bottom: 25px;
    }

    .voice-icon {
        font-size: 65px;
        margin-bottom: 15px;
    }

    .voice-title {
        color: white !important;
        font-size: 30px;
        font-weight: 800;
    }

    .voice-text {
        color: rgba(255,255,255,0.88) !important;
        font-size: 15px;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    div[data-testid="stMetric"] {
        background: white;
        border-radius: 15px;
        padding: 12px;
        border: 1px solid #e4ebf4;
    }


    /* ========================================================
       MAIN BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    div[data-testid="stChatInput"] {
        margin-bottom: 10px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 800px) {

        .main-container {
            padding: 10px 12px 20px 12px;
        }

        .user-bubble,
        .assistant-bubble {
            max-width: 95%;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "mode": "Chat",
    "persona": "Technical Interviewer",
    "voice_enabled": False,
    "messages": [],
    "documents": [],
    "selected_document": "All Documents",

    "interview_questions": [],
    "interview_answers": [],
    "interview_index": 0,
    "interview_active": False,
    "interview_finished": False,
    "evaluation": None,
    "question_count": 5,

    "last_voice_hash": None,
    "last_voice_text": None,
    "last_voice_response": None,
    "last_voice_audio": None,

    "interview_audio_hash": None,
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# OLLAMA
# ============================================================

@st.cache_resource
def get_llm():

    return Ollama(
        model="llama3.2:3b"
    )


llm = get_llm()


# ============================================================
# TEXT TO SPEECH
# ============================================================

def text_to_speech(text):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            165
        )

        engine.setProperty(
            "volume",
            1.0
        )

        temp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_path = temp.name

        temp.close()

        engine.save_to_file(
            text,
            temp_path
        )

        engine.runAndWait()

        engine.stop()

        if os.path.exists(temp_path):

            with open(
                temp_path,
                "rb"
            ) as file:

                audio = file.read()

            os.remove(temp_path)

            return audio

    except Exception:

        return None

    return None


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(uploaded_file):

    try:

        pdf_bytes = uploaded_file.getvalue()

        document_id = hashlib.sha256(
            pdf_bytes
        ).hexdigest()[:16]

        reader = PdfReader(
            io.BytesIO(pdf_bytes)
        )

        full_text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:

                full_text += page_text + "\n"

        if not full_text.strip():

            return False, "No readable text found."

        chunks = chunk_text(
            full_text
        )

        embeddings = create_embeddings(
            chunks
        )

        store_embeddings(
            chunks,
            embeddings,
            document_id,
            uploaded_file.name
        )

        if uploaded_file.name not in st.session_state.documents:

            st.session_state.documents.append(
                uploaded_file.name
            )

        return True, len(chunks)

    except Exception as e:

        return False, str(e)


# ============================================================
# RAG
# ============================================================

def retrieve_context(query):

    try:

        query_embedding = create_embeddings(
            [query]
        )[0]

        results = search(
            query_embedding,
            top_k=8
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        selected = st.session_state.selected_document

        context_parts = []

        for doc, metadata in zip(
            documents,
            metadatas
        ):

            filename = metadata.get(
                "filename",
                ""
            )

            if (
                selected != "All Documents"
                and filename != selected
            ):

                continue

            context_parts.append(
                f"[Source: {filename}]\n{doc}"
            )

        return "\n\n".join(
            context_parts
        )

    except Exception:

        return ""


# ============================================================
# CHAT RESPONSE
# ============================================================

def generate_chat_response(user_message):

    context = retrieve_context(
        user_message
    )

    persona = st.session_state.persona

    prompt = f"""
You are PersonaPrep AI.

Persona:
{persona}

You help the user with:
- General questions
- Projects
- Resume
- Interview preparation
- Uploaded documents
- Career preparation

Rules:

1. Answer naturally and clearly.
2. Use document context when relevant.
3. Do not invent information from documents.
4. If something is not present in the documents, say so.
5. You may answer general questions using your own knowledge.
6. Do not treat interviewer questions as candidate answers.
7. Do not invent candidate experiences.

DOCUMENT CONTEXT:

{context if context else "No relevant document context found."}

USER:

{user_message}

ANSWER:
"""

    try:

        return llm.invoke(
            prompt
        )

    except Exception as e:

        return f"Unable to generate response: {e}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🤖
            </div>

            <div class="brand-title">
                PersonaPrep AI
            </div>

            <div class="brand-subtitle">
                One AI. Every Perspective.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "＋  New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.mode = "Chat"

        st.session_state.interview_active = False

        st.session_state.interview_finished = False

        st.session_state.evaluation = None

        st.session_state.last_voice_hash = None

        st.rerun()


    # --------------------------------------------------------
    # MODES
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-heading">Modes</div>',
        unsafe_allow_html=True
    )


    mode_options = [
        "💬 Chat",
        "🎤 Interview",
        "📚 Documents",
        "🔊 Voice Chat"
    ]


    current_mode_display = None

    for option in mode_options:

        if option.endswith(
            st.session_state.mode
        ):

            current_mode_display = option

            break


    if current_mode_display is None:

        current_mode_display = "💬 Chat"


    selected_mode_display = st.radio(
        "Select mode",
        mode_options,
        index=mode_options.index(
            current_mode_display
        ),
        label_visibility="collapsed"
    )


    selected_mode = selected_mode_display.split(
        " ",
        1
    )[1]


    if selected_mode != st.session_state.mode:

        st.session_state.mode = selected_mode

        st.rerun()


    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # PERSONA
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-heading">🎯 Persona</div>',
        unsafe_allow_html=True
    )


    persona_options = [
        "Technical Interviewer",
        "HR Interviewer",
        "Friendly Mentor",
        "Strict Interviewer",
        "Career Coach",
        "Project Reviewer"
    ]


    st.session_state.persona = st.selectbox(
        "Persona",
        persona_options,
        index=persona_options.index(
            st.session_state.persona
        ),
        label_visibility="collapsed"
    )


    # --------------------------------------------------------
    # VOICE
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-heading">🔊 Voice</div>',
        unsafe_allow_html=True
    )


    st.session_state.voice_enabled = st.toggle(
        "Voice responses",
        value=st.session_state.voice_enabled
    )


    st.caption(
        "Microphone is available only in Voice Chat."
    )


    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    st.markdown(
        '<div class="side-heading">📚 Knowledge Base</div>',
        unsafe_allow_html=True
    )


    uploaded_files = st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )


    if uploaded_files:

        for uploaded_file in uploaded_files:

            if uploaded_file.name not in st.session_state.documents:

                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):

                    success, result = process_pdf(
                        uploaded_file
                    )

                if success:

                    st.success(
                        f"{uploaded_file.name} added"
                    )

                else:

                    st.error(
                        f"Could not process {uploaded_file.name}: {result}"
                    )


    # --------------------------------------------------------
    # DOCUMENT SELECTOR
    # --------------------------------------------------------

    if st.session_state.documents:

        options = [
            "All Documents"
        ] + st.session_state.documents


        if (
            st.session_state.selected_document
            not in options
        ):

            st.session_state.selected_document = (
                "All Documents"
            )


        st.session_state.selected_document = st.selectbox(
            "Document source",
            options
        )


        st.markdown(
            '<div class="side-heading">Uploaded PDFs</div>',
            unsafe_allow_html=True
        )


        for doc in st.session_state.documents:

            st.markdown(
                f"""
                <div class="document-pill">
                    📄 {doc}
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.caption(
            "No PDFs uploaded yet."
        )


    st.markdown(
        '<div class="side-divider"></div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    if st.button(
        "🗑  Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_voice_hash = None

        st.session_state.last_voice_text = None

        st.session_state.last_voice_response = None

        st.session_state.last_voice_audio = None

        st.rerun()


# ============================================================
# MAIN
# ============================================================

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="top-title">
        PersonaPrep AI
    </div>

    <div class="top-subtitle">
        One AI. Every Perspective.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT MODE
# ============================================================

if st.session_state.mode == "Chat":

    if not st.session_state.messages:

        st.markdown(
            """
            <div class="welcome-card">

                <div style="font-size:48px;">
                    🤖
                </div>

                <h1>
                    Welcome to PersonaPrep AI
                </h1>

                <p>
                    Ask questions, discuss your projects,
                    prepare for interviews and chat with your AI.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                '<div class="user-label">You</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="user-bubble">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="assistant-label">🤖 PersonaPrep AI</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="assistant-bubble">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if message.get("source"):

                st.markdown(
                    f"""
                    <div class="source-pill">
                        📄 {message["source"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # --------------------------------------------------------
    # TEXT CHAT ONLY
    # --------------------------------------------------------

    user_input = st.chat_input(
        "Message PersonaPrep AI..."
    )


    if user_input:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )


        with st.spinner(
            "PersonaPrep AI is thinking..."
        ):

            response = generate_chat_response(
                user_input
            )


        source = None

        if (
            st.session_state.selected_document
            != "All Documents"
        ):

            source = st.session_state.selected_document


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "source": source
            }
        )


        st.rerun()


# ============================================================
# DOCUMENTS MODE
# ============================================================

elif st.session_state.mode == "Documents":

    st.markdown(
        """
        <div class="mode-card">

            <div class="mode-title">
                📚 Knowledge Base
            </div>

            <div class="mode-description">
                Upload PDFs and use them as knowledge
                for PersonaPrep AI.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if st.session_state.documents:

        st.success(
            f"{len(st.session_state.documents)} document(s) available."
        )


        for doc in st.session_state.documents:

            st.markdown(
                f"📄 **{doc}**"
            )


        try:

            count = get_collection_count()

            st.info(
                f"Knowledge base contains {count} stored chunks."
            )

        except Exception:

            pass

    else:

        st.info(
            "Upload a PDF using the sidebar."
        )


# ============================================================
# INTERVIEW MODE
# ============================================================

elif st.session_state.mode == "Interview":

    st.markdown(
        """
        <div class="mode-card">

            <div class="mode-title">
                🎤 Interview Mode
            </div>

            <div class="mode-description">
                Practice an adaptive interview based on
                your selected persona and documents.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # START INTERVIEW
    # --------------------------------------------------------

    if not st.session_state.interview_active:

        question_count = st.slider(
            "Number of questions",
            3,
            15,
            st.session_state.question_count
        )


        st.session_state.question_count = (
            question_count
        )


        if st.button(
            "🚀 Start Interview",
            type="primary",
            use_container_width=True
        ):

            context = retrieve_context(
                "resume projects skills education experience technical knowledge"
            )


            prompt = f"""
Create {question_count} interview questions.

Persona:
{st.session_state.persona}

Use the candidate document context when available.

Questions should progress from basic
to moderate to difficult.

Return ONLY the questions,
one question per line.

Document context:

{context if context else "No document context available."}
"""


            try:

                raw_questions = llm.invoke(
                    prompt
                )


                questions = []


                for line in raw_questions.splitlines():

                    clean = re.sub(
                        r"^\s*(\d+[\).\:-]?|[-*])\s*",
                        "",
                        line
                    ).strip()


                    if clean:

                        questions.append(
                            clean
                        )


                questions = questions[
                    :question_count
                ]


                if not questions:

                    raise ValueError(
                        "No questions generated."
                    )


                st.session_state.interview_questions = (
                    questions
                )

                st.session_state.interview_answers = []

                st.session_state.interview_index = 0

                st.session_state.interview_active = True

                st.session_state.interview_finished = False

                st.session_state.evaluation = None

                st.rerun()


            except Exception as e:

                st.error(
                    f"Could not start interview: {e}"
                )


    # --------------------------------------------------------
    # ACTIVE INTERVIEW
    # --------------------------------------------------------

    else:

        questions = (
            st.session_state.interview_questions
        )

        index = (
            st.session_state.interview_index
        )

        total = len(
            questions
        )


        if index < total:

            st.progress(
                index / total,
                text=f"Question {index + 1} of {total}"
            )


            current_question = questions[index]


            st.markdown(
                f"""
                <div class="mode-card">

                    <div style="
                        font-size:13px;
                        color:#71839a;
                    ">
                        QUESTION {index + 1}
                    </div>

                    <div style="
                        font-size:22px;
                        font-weight:700;
                        color:#15365f;
                        margin-top:8px;
                    ">
                        {current_question}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # VOICE QUESTION
            # ------------------------------------------------

            if st.session_state.voice_enabled:

                if st.button(
                    "🔊 Read Question Aloud"
                ):

                    audio = text_to_speech(
                        current_question
                    )

                    if audio:

                        st.audio(
                            audio,
                            format="audio/wav"
                        )


            # ------------------------------------------------
            # ANSWER MODE
            # ------------------------------------------------

            answer_mode = st.radio(
                "Answer method",
                [
                    "⌨️ Type Answer",
                    "🎤 Voice Answer"
                ],
                horizontal=True
            )


            answer = None


            if answer_mode == "⌨️ Type Answer":

                answer = st.text_area(
                    "Your answer",
                    height=150,
                    placeholder="Type your answer here..."
                )


            else:

                st.info(
                    "Record your answer using the microphone."
                )


                audio_input = st.audio_input(
                    "🎤 Record your answer",
                    key=f"interview_audio_{index}"
                )


                if audio_input:

                    audio_bytes = (
                        audio_input.getvalue()
                    )


                    audio_hash = hashlib.sha256(
                        audio_bytes
                    ).hexdigest()


                    if (
                        st.session_state.interview_audio_hash
                        != audio_hash
                    ):

                        st.session_state.interview_audio_hash = (
                            audio_hash
                        )


                        try:

                            recognizer = (
                                sr.Recognizer()
                            )


                            with sr.AudioFile(
                                io.BytesIO(
                                    audio_bytes
                                )
                            ) as source:

                                audio_data = (
                                    recognizer.record(
                                        source
                                    )
                                )


                            answer = (
                                recognizer.recognize_google(
                                    audio_data
                                )
                            )


                            st.success(
                                f"Recognized answer: {answer}"
                            )


                        except Exception as e:

                            st.error(
                                f"Could not understand audio: {e}"
                            )


            # ------------------------------------------------
            # SUBMIT
            # ------------------------------------------------

            if st.button(
                "Submit Answer →",
                type="primary",
                use_container_width=True
            ):

                if answer and answer.strip():

                    st.session_state.interview_answers.append(
                        {
                            "question": current_question,
                            "answer": answer.strip()
                        }
                    )


                    st.session_state.interview_index += 1


                    st.rerun()

                else:

                    st.warning(
                        "Please provide an answer before submitting."
                    )


        else:

            st.session_state.interview_active = False

            st.session_state.interview_finished = True

            st.rerun()


# ============================================================
# INTERVIEW EVALUATION
# ============================================================

if (
    st.session_state.mode == "Interview"
    and st.session_state.interview_finished
    and st.session_state.evaluation is None
):

    answers = (
        st.session_state.interview_answers
    )


    if not answers:

        st.warning(
            "No candidate answers were provided, so an evaluation cannot be generated."
        )


    else:

        formatted_answers = ""


        for i, item in enumerate(
            answers,
            start=1
        ):

            formatted_answers += f"""
Question {i}:
{item["question"]}

Candidate Answer:
{item["answer"]}

"""


        evaluation_prompt = f"""
Evaluate this interview.

Persona:
{st.session_state.persona}

IMPORTANT:

Only evaluate the candidate's actual answers.

The interviewer questions are NOT candidate answers.

Do not invent:
- examples
- achievements
- weaknesses
- experiences
- statements

Score each category from 0 to 10:

Technical Knowledge
Communication
Problem Solving
Confidence
Overall Performance

Then provide:

1. Strengths
2. Areas for improvement
3. Detailed feedback
4. Interview summary

Candidate answers:

{formatted_answers}
"""


        with st.spinner(
            "Evaluating your interview..."
        ):

            try:

                evaluation = llm.invoke(
                    evaluation_prompt
                )

                st.session_state.evaluation = (
                    evaluation
                )

            except Exception as e:

                st.error(
                    f"Evaluation failed: {e}"
                )


# ============================================================
# SHOW EVALUATION
# ============================================================

if (
    st.session_state.mode == "Interview"
    and st.session_state.evaluation
):

    evaluation = (
        st.session_state.evaluation
    )


    st.markdown(
        """
        <div class="mode-card">

            <div class="mode-title">
                📊 Interview Evaluation
            </div>

            <div class="mode-description">
                Your performance analysis from PersonaPrep AI.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    def extract_score(
        label,
        text
    ):

        pattern = (
            rf"{re.escape(label)}"
            r"\s*[:\-]?\s*"
            r"(\d+(?:\.\d+)?)"
            r"\s*(?:/10)?"
        )


        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )


        if match:

            return float(
                match.group(1)
            )


        return 0.0


    technical = extract_score(
        "Technical Knowledge",
        evaluation
    )


    communication = extract_score(
        "Communication",
        evaluation
    )


    problem_solving = extract_score(
        "Problem Solving",
        evaluation
    )


    confidence = extract_score(
        "Confidence",
        evaluation
    )


    overall = extract_score(
        "Overall Performance",
        evaluation
    )


    cols = st.columns(5)


    scores = [
        ("Technical", technical),
        ("Communication", communication),
        ("Problem Solving", problem_solving),
        ("Confidence", confidence),
        ("Overall", overall)
    ]


    for col, (label, score) in zip(
        cols,
        scores
    ):

        with col:

            st.metric(
                label,
                f"{score:.1f}/10"
            )


    if overall:

        st.progress(
            min(
                max(
                    int(overall * 10),
                    0
                ),
                100
            ),
            text=f"Overall Performance: {overall:.1f}/10"
        )


    st.markdown(
        "### 📈 Performance"
    )


    chart_data = {
        "Technical Knowledge": technical,
        "Communication": communication,
        "Problem Solving": problem_solving,
        "Confidence": confidence,
        "Overall": overall
    }


    chart_df = pd.DataFrame(
        chart_data,
        index=["Score"]
    ).T


    st.bar_chart(
        chart_df
    )


    st.markdown(
        "### 📝 Detailed Feedback"
    )


    st.markdown(
        evaluation
    )


    if st.button(
        "🔄 Start New Interview",
        type="primary",
        use_container_width=True
    ):

        st.session_state.interview_questions = []

        st.session_state.interview_answers = []

        st.session_state.interview_index = 0

        st.session_state.interview_active = False

        st.session_state.interview_finished = False

        st.session_state.evaluation = None

        st.rerun()


# ============================================================
# VOICE CHAT
# ============================================================

elif st.session_state.mode == "Voice Chat":

    st.markdown(
        """
        <div class="voice-screen">

            <div class="voice-icon">
                🎙️
            </div>

            <div class="voice-title">
                Voice Chat
            </div>

            <div class="voice-text">
                Talk naturally with PersonaPrep AI.
                Your microphone is active only in this mode.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    audio_input = st.audio_input(
        "🎤 Tap here and speak",
        key="voice_chat_input"
    )


    if audio_input:

        audio_bytes = (
            audio_input.getvalue()
        )


        audio_hash = hashlib.sha256(
            audio_bytes
        ).hexdigest()


        if (
            st.session_state.last_voice_hash
            != audio_hash
        ):

            st.session_state.last_voice_hash = (
                audio_hash
            )


            try:

                recognizer = sr.Recognizer()


                with sr.AudioFile(
                    io.BytesIO(
                        audio_bytes
                    )
                ) as source:

                    audio_data = (
                        recognizer.record(
                            source
                        )
                    )


                with st.spinner(
                    "Listening..."
                ):

                    transcript = (
                        recognizer.recognize_google(
                            audio_data
                        )
                    )


                st.session_state.last_voice_text = (
                    transcript
                )


                st.markdown(
                    "### 🗣️ You said"
                )


                st.info(
                    transcript
                )


                with st.spinner(
                    "PersonaPrep AI is thinking..."
                ):

                    response = (
                        generate_chat_response(
                            transcript
                        )
                    )


                st.session_state.last_voice_response = (
                    response
                )


                st.markdown(
                    "### 🤖 PersonaPrep AI"
                )


                st.markdown(
                    response
                )


                if st.session_state.voice_enabled:

                    audio = text_to_speech(
                        response
                    )


                    if audio:

                        st.session_state.last_voice_audio = (
                            audio
                        )


                        st.audio(
                            audio,
                            format="audio/wav"
                        )


            except sr.UnknownValueError:

                st.error(
                    "I couldn't understand the audio. Please try again."
                )


            except Exception as e:

                st.error(
                    f"Voice processing failed: {e}"
                )


# ============================================================
# CLOSE MAIN CONTAINER
# ============================================================

st.markdown(
    "</div>",
    unsafe_allow_html=True
)