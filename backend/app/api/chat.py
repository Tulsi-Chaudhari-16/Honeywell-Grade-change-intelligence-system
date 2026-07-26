from fastapi import APIRouter
from pydantic import BaseModel
from app.core.llm_factory import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    machine_id: str | None = "PM1"

@router.post("")
async def chat_with_factory(payload: ChatRequest):
    llm = get_llm(temperature=0.7)
    sys_msg = SystemMessage(
        content="You are GCIS (Grade Change Intelligence System). "
                "You are an expert AI assistant helping operators at a paper mill. "
                "Keep answers brief, highly technical but accessible to operators."
    )
    user_msg = HumanMessage(content=payload.message)
    
    try:
        response = await llm.ainvoke([sys_msg, user_msg])
        return {"reply": response.content}
    except Exception as e:
        return {"reply": f"Error communicating with LLM: {str(e)}"}
