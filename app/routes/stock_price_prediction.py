from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from services import analysis


router = APIRouter()

class StateScheme(BaseModel):
    code: str

@router.get("/")
async def prediction(code:str):
    # 分析情報を取得
    result = analysis.get_prediction(code)
    return result
    

    
