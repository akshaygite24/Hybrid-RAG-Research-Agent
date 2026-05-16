import streamlit as st
import time
from pipeline import run_pipeline
from rag.ingest import ingest_pdf
from langchain_core.messages import HumanMessage, AIMessage
import tempfile
import os
import logging
import traceback

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="Research Agent",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


with st.sidebar:
    st.markdown("Research Agent")
    st.markdown("----")

    st.markdown("#### Upload Document")
    uploaded_file = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")

    if uploaded_file and uploaded_file.name != st.session_state.pdf_name:
        with st.spinner("Ingesting PDF...."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                ingest_pdf(tmp_path)
                os.unlink(tmp_path)

                st.session_state.pdf_ingested = True
                st.session_state.pdf_name = uploaded_file.name
                st.success(f"Ingested {uploaded_file.name}")

            except Exception as e:
                st.error(f"Failed to ingest PDF: {str(e)}")
    
    if st.session_state.pdf_ingested:
        st.markdown(f"Active Document {st.session_state.pdf_name}")

    st.markdown("----")

    st.markdown("Settings")
    use_planner = st.toggle("Use Planner", value=True, help="Break query into sub-questions")
    use_critic = st.toggle("Use Critic", value=True, help="Review and Improve final answer")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("----")
    st.markdown("Pipeline")
    st.markdown("""
    1. Planner -> sub-questions
    2. Agent -> tool routing
    3. RAG -> PDF search
    4. Web -> Live search
    5. Critic -> Improvement
    """)

st.markdown("Hybrid RAG Research Agent")
st.markdown("----")

for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])

if query := st.chat_input("Ask Anything - I'll search you documents and the web...."):
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.status("Running research pipeline....", expanded=True) as status:
            st.write("Thinking!")
            time.sleep(0.5)

            try:
                result = run_pipeline(
                    query=query,
                    chat_history=st.session_state.chat_history,
                    use_planner=use_planner,
                    use_critic=use_critic
                )

                if result["used_planner"]:
                    st.write("Planner breaking down query....")
                    if len(result["sub_questions"]) > 1:
                        st.write("Sub-questions identified:")
                        for q in result["sub_questions"]:
                            st.write(f"{q}")
                
                st.write("Researching with tools....")
                if result["used_critic"]:
                    st.write("Critic reviewing answer....")
                status.update(label="Done!", state="complete", expanded=False)

            except Exception as e:
                error_details = traceback.format_exc()
                status.update(label="Failed!", state="error")
                st.error(f"Pipeline error: {str(e)}")
                st.code(error_details)  # shows full traceback in UI
                result = None
                    
        if result:
            st.markdown(result["final_answer"])

            st.session_state.chat_history.append(HumanMessage(content=query))
            st.session_state.chat_history.append(AIMessage(content=result["final_answer"]))
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["final_answer"]
            })
            