from langgraph.graph import MessagesState # type: ignore
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage , RemoveMessage # type: ignore
from pydantic import BaseModel, Field # type: ignore
from langchain_groq import ChatGroq  # type: ignore #*********************************************************************
from typing import Literal 
from langchain.schema import Document  

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved historical documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GraderNode:
    def __init__(self, model_name: str):
        
        model = ChatGroq(model_name=model_name, temperature=0)
        self.structured_llm_grader = model.with_structured_output(GradeDocuments)

        self.grade_prompt ="""You are a teacher grading a quiz. You will be given: 
        1/ a QUESTION
        2/ A FACT provided by the student
        
        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
        1 is the highest (best) score. 0 is the lowest score you can give. 
        
        Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
        
        Avoid simply stating the correct answer at the outset.
        
        Question: {question} \n
        Fact: \n\n {documents} \n\n
        
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        """ 

    def grade(self, question: str, doc_txt: str):
        result = self.structured_llm_grader.invoke([SystemMessage(content=self.grade_prompt.format(question=question,documents=doc_txt))])
        return result.binary_score
        
class GenerationNode:
    
    def __init__(self,model_name: str):
        
        self.model = ChatGroq(model_name=model_name, temperature=0)
        self.generation_prompt =   """You are an assistant for question-answering tasks (specifically syrian heritage). 
            You've got a summary of your current conversation (it may be empty): {memory}.
            
            Use the following documents (from the database/web search using tavily and duckduckgo) to augment your internal knowledge to answer to the question. 
            
            - If you don't know the answer, just say that you don't know. 
            - If the user asks a simple question, just answer politely (such as greeting).
            - Do not discuss anything outside the scope of syrian heritage, chat about syrian history and only syrian history.
            - Answer in the language of the question.
            
            Question: {question} 
            Documents: {documents} 
            Answer:  """
        
    def generate(self, question: str , documents: str, memory ):
        result = self.model.invoke([SystemMessage(content=self.generation_prompt.format(question=question,documents=documents,memory=memory))])
        return result.content
        
    


