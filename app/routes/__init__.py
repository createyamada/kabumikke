from fastapi import APIRouter
from routes.auth import router as auth
from routes.stock_price_prediction import router as stock_price_prediction


router = APIRouter()
router.include_router(auth, prefix="/auth", tags=["auth"])
router.include_router(stock_price_prediction, prefix="/stock_price_prediction", tags=["stock_price_prediction"])
