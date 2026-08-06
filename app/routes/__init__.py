from fastapi import APIRouter, Depends
from routes.auth import router as auth
from routes.auth import require_authenticated
from routes.stock_price_prediction import router as stock_price_prediction
from routes.prime_ranking import router as prime_ranking


router = APIRouter()
router.include_router(auth, prefix="/auth", tags=["auth"])
router.include_router(
    stock_price_prediction,
    prefix="/stock_price_prediction",
    tags=["stock_price_prediction"],
    dependencies=[Depends(require_authenticated)],
)
router.include_router(
    prime_ranking,
    prefix="/prime-ranking",
    tags=["prime-ranking"],
    dependencies=[Depends(require_authenticated)],
)
