# ========== Import Modules ==========
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.tools import Tool, tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from app.helper import *
from app.tools import geocode_address_tool, get_traffic_status_tool, get_weather_tool, get_current_time_tool

# ========== Configuration ==========
LLM_NAME = "qwen-qwq-32b"
VECTOR_STORE_DIR = "./chroma_db"
COLLECTION_NAME = "agent_collection"

# ========== Initialize Components ==========

# Initialize LLM
llm = init_chat_model(LLM_NAME, model_provider="groq")

# Initialize embeddings
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
# embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
# embedding = SentenceTransformer("thenlper/gte-large")


# Initialize vector store
# vector_store = Chroma(
#     collection_name=COLLECTION_NAME,
#     embedding_function=embedding,
#     persist_directory=VECTOR_STORE_DIR
# )

# vector_store = initialize_db("./app/traffic_data", embedding)

# ========== Document Processing ==========
def add_pdf_to_vectorstore(pdf_path: str) -> None:
    pass
#     """Load PDF, split into chunks, and add to vector store."""
#     documents = PyPDFLoader(pdf_path).load()
#     chunks = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     ).split_documents(documents)
#     vector_store.add_documents(chunks)

# ========== Tools ==========
# class RetrieveDocumentsInput(BaseModel):
#     query: str

# @tool
# def retrieve_documents(query: str) -> str:
#     """Search vector store and return top results."""
#     results = vector_store.similarity_search(query, k=5)
#     content = "\n\n".join(
#         f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in results
#     )
#     return content

# Wrap tool
# retrieve_documents_tool = Tool(
#     name="retrieve_documents",
#     description="Retrieve information related to user's query. Parameters include query.",
#     func=retrieve_documents,
#     args_schema=RetrieveDocumentsInput
# )

search_tool = DuckDuckGoSearchRun()


# ========== LangGraph Nodes ==========

# def query_or_respond(state: MessagesState):
#     """Generate tool call or direct response."""
#     llm_with_tools = llm.bind_tools([retrieve_documents_tool])
#     response = llm_with_tools.invoke(state["messages"])
#     return {"messages": [response]}

# def generate(state: MessagesState):
#     """Use retrieved content to generate a final response."""
#     # Extract recent tool messages
#     tool_messages = []
#     for msg in reversed(state["messages"]):
#         if msg.type == "tool":
#             tool_messages.insert(0, msg)
#         else:
#             break

#     context = "\n\n".join(msg.content for msg in tool_messages)

#     system_prompt = (
#         # "You are an assistant for question-answering tasks. "
#         # "Use the following pieces of retrieved context to answer the question. "
#         # "If you don't know the answer, say that you don't see it in file and answer by your knowledges. "
#         # "Use 2000 sentences maximum and keep the answer detailed."
#         "You are a smart and multilingual assistant for answering user questions in their native language. Depending on the question, you may:"
#         "1. Answer immediately if you already know the answer."
#         "2. If you do not know the answer, attempt to retrieve information from the provided documents."
#         "3. If the documents are missing the necessary information or are insufficient, search the web for the most accurate and up-to-date response."

#         "You must:"

#         "- Answer using the language of the user’s question."
#         "- Provide a detailed answer, using a maximum of 2000 sentences."
#         "- Clearly specify which parts of the answer come from retrieved documents and which parts come from web search."
#         "- If none of the sources provide sufficient data, explain that and give the best possible answer based on general knowledge."

#         "Your responses must be highly detailed, precise, and using english."
#         f"\n\n{context}"
#     )

#     conversation = [
#         msg for msg in state["messages"]
#         if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
#     ]

#     prompt = [SystemMessage(system_prompt)] + conversation
#     response = llm.invoke(prompt)

#     return {"messages": [response]}

# ========== LangGraph Construction ==========

# tools = [retrieve_documents]
# tool_node = ToolNode(tools)

# graph_builder = StateGraph(MessagesState)
# graph_builder.add_node("query_or_respond", query_or_respond)
# graph_builder.add_node("tool_node", tool_node)
# graph_builder.add_node("generate", generate)

# graph_builder.set_entry_point("query_or_respond")
# graph_builder.add_edge(START, "query_or_respond")
# # graph_builder.add_edge("query_or_respond", "tool_node")
# graph_builder.add_conditional_edges(
#     "query_or_respond",
#     tools_condition,
#     {END: END, "tools": "tool_node"}
# )
# graph_builder.add_edge("tool_node", "generate")
# graph_builder.add_edge("generate", END)

# # Compile the graph into a runnable model
# agent_model = graph_builder.compile()

# tools = [retrieve_documents_tool, search_tool, geocode_address_tool, get_traffic_status_tool, get_weather_tool]
tools = [search_tool, geocode_address_tool, get_traffic_status_tool, get_weather_tool, get_current_time_tool]
agent_model = create_react_agent(llm, tools)

# ========== Querry ==========
def query_llm(prompt: str) -> str:
    """Helper for sending a user message to the model with a system prompt."""
    system_prompt = (
        "You are a smart and multilingual assistant for answering user questions in traffic infomation. "
        "Depending on the question, you may:\n\n"
        "1. Answer immediately if you already know the answer.\n"
        "2. Depending on user's question, you need to use suitable tool or combine tools to answer\n\n"
        "You must:\n"
        "- Answer using Vietnamese language\n"
        "- Provide a detailed answer, using a maximum of 2000 sentences.\n"
        "- If none of the sources provide sufficient data, explain that and give the best possible answer based on general knowledge."
    )

    response = agent_model.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
    })

    return response['messages'][-1].content
