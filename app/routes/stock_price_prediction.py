from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from services import analysis
from services import edinet
from services.cross_sectional import fetch_and_run_cross_sectional_backtest


router = APIRouter()

class StateScheme(BaseModel):
    code: str

@router.get("/")
async def prediction(code:str):
    # 分析情報を取得
    result = analysis.get_prediction(code)
    return result


@router.get("/cross-sectional-backtest")
async def cross_sectional_backtest(codes: str, period: str = "5y", top_n: int = 10, rebalance_days: int = 5):
    code_list = [code.strip() for code in codes.split(",") if code.strip()]
    if len(code_list) < 2 or any(not code.isdigit() or len(code) != 4 for code in code_list):
        raise HTTPException(status_code=422, detail="codes must contain at least two comma-separated 4-digit codes")
    if len(code_list) > 200:
        raise HTTPException(status_code=422, detail="a maximum of 200 codes is allowed per request")
    return fetch_and_run_cross_sectional_backtest(code_list, period, top_n, rebalance_days)


@router.get("/fundamentals")
async def fundamentals(code: str):
    if not code.isdigit() or len(code) != 4:
        raise HTTPException(status_code=422, detail="code must be a 4-digit stock code")
    return edinet.get_fundamental_analysis(code)
    

    
