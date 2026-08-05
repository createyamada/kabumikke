from fastapi import APIRouter, HTTPException

from services import prime_ranking


router = APIRouter()


@router.get("/")
async def latest_ranking(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    return prime_ranking.read_latest_ranking(limit)


@router.get("/status")
async def ranking_status():
    return prime_ranking.read_status()


@router.post("/refresh", status_code=202)
async def refresh_ranking(limit: int = 10, shortlist_size: int = 50):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    if shortlist_size < limit or shortlist_size > 200:
        raise HTTPException(status_code=422, detail="shortlist_size must be between limit and 200")
    return prime_ranking.start_prime_ranking_refresh(limit, shortlist_size)
