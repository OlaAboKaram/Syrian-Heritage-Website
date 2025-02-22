from functions import *
from langgraph.graph import StateGraph, END , START
from langgraph.checkpoint.memory import MemorySaver


def build_graph(): 
    workflow = StateGraph(State)
    workflow.add_node('intent_classifier', intent_classifier)
    workflow.add_node("retrieve", retrieve)  
    workflow.add_node("grade_documents", grade_documents)  
    workflow.add_node("generate", generate)  
    #workflow.add_node("web_search", web_search) 
    workflow.add_node('tavily_search',tavily_search)
    workflow.add_node('duckduckgo_search',duckduckgo_search)
    workflow.add_node("summarize_conversation", summarize_conversation) 

    workflow.add_edge(START, "intent_classifier")
    workflow.add_edge('summarize_conversation', END)
    workflow.add_edge('retrieve','grade_documents')

    #workflow.add_edge("retrieve", "grade_documents")

    # workflow.add_conditional_edges(
    #     "grade_documents",
    #     decide_to_generate,
    #     {
    #         "search": "web_search",
    #         "generate": "generate",
    #     },
    # )
    # workflow.add_conditional_edges(
    #     "grade_documents",
    #     decide_to_generate,
    #     {
    #         "search": "tavily_search",
    #         "generate": "generate",
    #     },
    # )
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generateV2,
    )
    workflow.add_conditional_edges(
        "generate",
        summarization_router,
        {
            'summarize_conversation': "summarize_conversation",
            'END': END,
        },
    )
    workflow.add_conditional_edges(
        "intent_classifier",
        no_rag_router,
        {
            'retrieve': "retrieve",
            'generate': 'generate',
        },
    )
    #workflow.add_edge("web_search", "generate")
    workflow.add_edge("tavily_search", "generate")
    workflow.add_edge("duckduckgo_search", "generate")
    memory = MemorySaver()
    graph = workflow.compile()
    return graph
