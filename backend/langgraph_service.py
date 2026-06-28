from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from typing import TypedDict

try:
    from .config import Config
except ImportError:
    from config import Config


@lru_cache(maxsize=1)
def get_llm():
    if not Config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return ChatGroq(
        groq_api_key=Config.GROQ_API_KEY,
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )


class State(TypedDict):
    messages: list[HumanMessage | AIMessage | SystemMessage]


SYSTEM_PROMPT = (
    "You are PropAI, an expert real estate intelligence assistant. "
    "You help users with property search, market analysis, investment advice, "
    "legal document Q&A, lead qualification, and client interaction. "
    "When recommending properties, ask clarifying questions about budget, location, and preferences. "
    "Be professional, concise, and helpful. Do not repeat your previous response verbatim. "
    "If anyone gives a document that is not related to real estate, you should read the document in more detail and give full information to user, and you should not say you need more document information. "
    "Don't hesitate to provide detailed information from the document, even if it's not directly related to real estate. Your goal is to be as informative and helpful as possible based on the provided context."
)


def call_model(state):
    history = state["messages"]
    model = get_llm()
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT), *history])

    last_ai = next(
        (m for m in reversed(history) if isinstance(m, AIMessage)), None
    )

    if last_ai and response.content.strip() == last_ai.content.strip():
        response = model.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
            SystemMessage(content="Rephrase and improve your answer. Add new detail and avoid repetition."),
        ])

    return {"messages": history + [response]}


graph = StateGraph(State)
graph.add_node("chatbot", call_model)
graph.set_entry_point("chatbot")
app_graph = graph.compile()


def get_response(history):
    result = app_graph.invoke({"messages": history})
    return result["messages"][-1]