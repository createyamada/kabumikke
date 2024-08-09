from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from services import analysis


router = APIRouter()

class StateScheme(BaseModel):
    code: str

@router.get("/")
# curl -X GET http://localhost:8888/api/stock_price_prediction/?code=9984 -H 'Content-Type: application/json'
async def prediction(code:str):
    # 分析情報を取得
    result = analysis.get_prediction(code)
    return result
    

    
