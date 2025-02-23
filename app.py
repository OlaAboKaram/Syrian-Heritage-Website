from typing import Optional
from rich import print as rprint
from rich.panel import Panel
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from graph import build_graph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, RemoveMessage  # type: ignore

graph = build_graph()

config = {"configurable": {"thread_id": "2"}}

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    summary: str
    session_id: int 
    user_id: Optional[int] = ''

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    print('Received user input...')

    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    human_message = HumanMessage(content=user_input)
    
    try:
        output = graph.invoke({"messages": [human_message], "question": user_input, "summary": request.summary}, config)
        response_text = str(output['generation'])
        new_summary = str(output['summary'])

        rprint(Panel("AI: " + response_text))

        return {
            "message": response_text,
            "session_id": session_id,
            "summary": new_summary,
            "user_id": user_id  
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
