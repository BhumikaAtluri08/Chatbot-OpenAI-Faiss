from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import chat_agent_executor

# Load vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore", embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="lookup_product_info",
    description="Useful for answering questions about laser or optics products."
)

# Setup LLM
llm = ChatOpenAI(model="gpt-4o")

# Memory
memory = ConversationBufferMemory(return_messages=True)

# Define system prompt as a simple string
prompt = "You are SalesBot, an expert in Coherent product documentation."

tools = [retriever_tool]

# Create the agent and executor
agent_executor = chat_agent_executor.create_tool_calling_executor(
    tools=tools,
    prompt=prompt,
    model=llm
)

# Export the agent_executor as salesbot_executor for import in chat_ui.py
salesbot_executor = agent_executor

# Main function to run SalesBot
def run_salesbot(question: str, chat_history: list):
    messages = chat_history + [HumanMessage(content=question)]
    result = agent_executor.invoke({"input": question, "chat_history": chat_history}, config=RunnableConfig())
    answer = result["output"]
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history
