import streamlit as st
import ollama
import hashlib
import os

from loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import create_embeddings
from rag.vector_store import store_embeddings, search


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="PersonaPrep AI",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🤖 PersonaPrep AI")
st.subheader("One AI. Every Perspective.")


# ==================================================
# PERSONAS
# ==================================================

personas = {
    "Faculty": """
You are a college faculty member.
Ask academic and technical questions.
Be professional and slightly strict.
Evaluate the student's understanding.
""",

    "Interviewer": """
You are a professional job interviewer.
Ask realistic interview questions.
Ask follow-up questions based on the student's answers.
Evaluate technical knowledge, communication and problem solving.
""",

    "Manager": """
You are a professional manager.
Ask workplace, teamwork and problem-solving questions.
Focus on decision making and communication.
""",

    "HR": """
You are an HR interviewer.
Ask behavioral, teamwork, communication and career questions.
Be professional and realistic.
""",

    "Friend": """
You are a friendly person having a natural conversation.
Be casual, supportive and conversational.
""",

    "Mentor": """
You are an experienced career mentor.
Ask thoughtful questions and help the student improve.
Give useful career-oriented feedback.
"""
}


# ==================================================
# PERSONA SELECTION
# ==================================================

persona = st.selectbox(
    "🎭 Choose your AI Persona",
    list(personas.keys())
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📚 Document Manager")


uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ==================================================
# SESSION STATE
# ==================================================

if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = {}


if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# PROCESS DOCUMENTS
# ==================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        file_bytes = uploaded_file.getvalue()

        document_id = hashlib.sha256(
            file_bytes
        ).hexdigest()[:16]

        if document_id not in st.session_state.processed_documents:

            with st.spinner(
                f"📚 Processing {uploaded_file.name}..."
            ):

                os.makedirs(
                    "data/documents",
                    exist_ok=True
                )

                pdf_path = os.path.join(
                    "data",
                    "documents",
                    f"{document_id}.pdf"
                )

                with open(
                    pdf_path,
                    "wb"
                ) as f:
                    f.write(file_bytes)

                document_text = load_document(
                    pdf_path
                )

                if not document_text.strip():

                    st.error(
                        f"❌ Could not extract text from "
                        f"{uploaded_file.name}"
                    )

                    continue

                document_chunks = chunk_text(
                    document_text
                )

                document_embeddings = create_embeddings(
                    document_chunks
                )

                # IMPORTANT:
                # Four arguments are required here.
                store_embeddings(
                    document_chunks,
                    document_embeddings,
                    document_id,
                    uploaded_file.name
                )

                st.session_state.processed_documents[
                    document_id
                ] = {
                    "filename": uploaded_file.name,
                    "document_id": document_id,
                    "chunks": len(document_chunks)
                }

            st.sidebar.success(
                f"✅ {uploaded_file.name}"
            )


# ==================================================
# DOCUMENT SELECTION
# ==================================================

if st.session_state.processed_documents:

    st.sidebar.divider()

    st.sidebar.subheader(
        "🔎 Search Documents"
    )

    document_options = {
        "All Documents": None
    }

    for document_id, document_info in (
        st.session_state.processed_documents.items()
    ):

        document_options[
            document_info["filename"]
        ] = document_id

    selected_document = st.sidebar.selectbox(
        "Search in:",
        list(document_options.keys())
    )

    selected_document_id = document_options[
        selected_document
    ]

    st.sidebar.divider()

    st.sidebar.subheader(
        "📄 Uploaded Documents"
    )

    for document_info in (
        st.session_state.processed_documents.values()
    ):

        st.sidebar.write(
            f"• {document_info['filename']}"
        )

        st.sidebar.caption(
            f"{document_info['chunks']} chunks"
        )

else:

    selected_document_id = None


# ==================================================
# DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# ==================================================
# CHAT INPUT
# ==================================================

user_input = st.chat_input(
    "💬 Ask something about your documents..."
)


# ==================================================
# HANDLE USER QUESTION
# ==================================================

if user_input:

    if not st.session_state.processed_documents:

        st.warning(
            "📄 Please upload at least one PDF first."
        )

        st.stop()

    # ------------------------------------------------
    # Create query embedding
    # ------------------------------------------------

    with st.spinner(
        "🔎 Searching your documents..."
    ):

        query_embedding = create_embeddings(
            [user_input]
        )[0]

        results = search(
            query_embedding,
            top_k=3,
            document_id=selected_document_id
        )

    # ------------------------------------------------
    # Get relevant chunks
    # ------------------------------------------------

    relevant_chunks = results[
        "documents"
    ][0]

    # ------------------------------------------------
    # Get source files
    # ------------------------------------------------

    source_files = set()

    if results.get("metadatas"):

        for metadata in results["metadatas"][0]:

            if metadata:

                filename = metadata.get(
                    "filename"
                )

                if filename:
                    source_files.add(filename)

    # ------------------------------------------------
    # Build context
    # ------------------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )

    # ------------------------------------------------
    # Save user message
    # ------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # ------------------------------------------------
    # System prompt
    # ------------------------------------------------

    system_prompt = f"""
You are PersonaPrep AI.

Your current persona is:

{personas[persona]}

Use the following information retrieved
from the user's uploaded documents:

{context}

Important rules:

- Use the retrieved documents as your main source.
- Do not invent information.
- If the answer is not available, clearly say so.
- Stay in character as the selected persona.
- Ask one question at a time during interviews.
- Adapt follow-up questions based on previous answers.
- Do not repeatedly ask the same question.
- Keep the conversation natural and useful.
"""

    # ------------------------------------------------
    # Prepare messages for Ollama
    # ------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(
        st.session_state.messages
    )

    # ------------------------------------------------
    # Call Ollama
    # ------------------------------------------------

    with st.spinner(
        "🤖 PersonaPrep AI is thinking..."
    ):

        response = ollama.chat(
            model="llama3.2:3b",
            messages=messages
        )

    ai_reply = response[
        "message"
    ][
        "content"
    ]

    # ------------------------------------------------
    # Add source information
    # ------------------------------------------------

    if source_files:

        ai_reply += (
            "\n\n📚 **Sources:** "
            + ", ".join(
                sorted(source_files)
            )
        )

    # ------------------------------------------------
    # Save AI response
    # ------------------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_reply
    })

    # ------------------------------------------------
    # Refresh
    # ------------------------------------------------

    st.rerun()