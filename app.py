from rich import print as rprint
from rich.panel import Panel
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from graph import build_graph
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage , RemoveMessage # type: ignore

graph = build_graph()

config = {"configurable": {"thread_id": "2"}}

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    summary: str
    session_id: str
    user_id: str

# while True:
#     user_input = input('You: ')
#     if user_input.lower() == 'exit':
#         break
#     human_message = HumanMessage(content=user_input)
#     output = graph.invoke({"messages": [human_message] , 'question': user_input}, config) 
#     rprint(Panel("AI: " + str(output['generation'])))


@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    print('Recieved user inpit....')
    if not user_input.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    human_message = HumanMessage(content=user_input)
    
    try:
        
        output = graph.invoke({"messages": [human_message], "question": user_input , "summary": request.summary}, config)
        response_text = str(output['generation'])
        new_summary = str(output['summary'])

        rprint(Panel("AI: " + response_text))

        return {"message": response_text,'session_id':request.session_id , 'summary': new_summary , 'user_id': request.user_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
