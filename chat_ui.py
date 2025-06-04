import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.salesbot_agent import salesbot_executor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import shutil
import io
from fpdf import FPDF

st.set_page_config(page_title="SalesBot", layout="wide")
st.title("🤖 SalesBot: Ask About Product Docs")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display existing chat messages
for i, chat in enumerate(st.session_state.chat_history):
    role, message = chat["role"], chat["content"]
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

# User input
user_query = st.chat_input("Ask a question about the products...")
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("SalesBot is thinking..."):
        # Build messages list for the agent
        messages = []
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                from langchain_core.messages import HumanMessage
                messages.append(HumanMessage(content=chat["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=chat["content"]))
        # Add the current user query
        from langchain_core.messages import HumanMessage
        messages.append(HumanMessage(content=user_query))
        response = salesbot_executor.invoke({"messages": messages})
        print("RAW RESPONSE:", response)  # Debug: print the raw response to the terminal
        import logging
        logging.warning(f"RAW RESPONSE: {response}")
        # Try to extract the reply from the response
        reply = response.get("output")
        if not reply:
            # Try to extract from the last AIMessage in 'messages'
            ai_messages = [m for m in response.get("messages", []) if getattr(m, 'type', None) == 'ai']
            if ai_messages:
                reply = ai_messages[-1].content
            else:
                reply = "Sorry, I couldn’t find anything."
        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

with st.sidebar:
    st.header("📄 Upload Product PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        if st.button("Ingest PDFs"):
            with st.spinner("Processing PDFs..."):
                temp_dir = tempfile.mkdtemp()
                local_paths = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    local_paths.append(file_path)

                # Load and split docs
                docs = []
                for path in local_paths:
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)

                # Embed and update vectorstore
                from agents.salesbot_agent import embedding, db, salesbot_executor, create_retriever_tool
                new_db = FAISS.from_documents(chunks, embedding)
                db.merge_from(new_db)

                # Update retriever and tool
                from agents.salesbot_agent import create_retriever_tool, llm, prompt
                retriever = db.as_retriever(search_kwargs={"k": 3})
                new_retriever_tool = create_retriever_tool(
                    retriever,
                    name="lookup_product_info",
                    description="Useful for answering questions about laser or optics products."
                )
                from langgraph.prebuilt import chat_agent_executor
                new_tools = [new_retriever_tool]
                new_executor = chat_agent_executor.create_tool_calling_executor(
                    tools=new_tools,
                    prompt=prompt,
                    model=llm
                )
                # Replace the global executor for future queries
                import agents.salesbot_agent as salesbot_agent
                salesbot_agent.salesbot_executor = new_executor

                shutil.rmtree(temp_dir)
                st.success("✅ PDFs ingested and retriever updated.")

# Add a button to download chat as PDF
if st.session_state.chat_history:
    def generate_pdf(chat_history):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "SalesBot Conversation Report", ln=True, align="C")
        pdf.ln(10)
        for chat in chat_history:
            role = "User" if chat["role"] == "user" else "SalesBot"
            pdf.set_font("Arial", style="B", size=12)
            pdf.cell(0, 10, f"{role}:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, chat["content"])
            pdf.ln(2)
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return io.BytesIO(pdf_bytes)

    pdf_bytes = generate_pdf(st.session_state.chat_history)
    st.download_button(
        label="Download Conversation as PDF",
        data=pdf_bytes,
        file_name="salesbot_conversation.pdf",
        mime="application/pdf"
    )