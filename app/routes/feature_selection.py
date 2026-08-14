from typing import Optional

from fastapi import APIRouter, Query

from services import feature_selection


router = APIRouter()


@router.get("/")
def get_feature_selection_dashboard(
    code: Optional[str] = None,
    sector: Optional[str] = None,
    model: Optional[str] = None,
    horizon: Optional[int] = None,
    regime: Optional[str] = None,
    status: Optional[str] = None,
    minimum_evaluations: int = Query(0, ge=0),
):
    return feature_selection.dashboard(
        code=code,
        sector=sector,
        model=model,
        horizon=horizon,
        regime=regime,
        status=status,
        minimum_evaluations=minimum_evaluations,
    )


@router.get("/events")
def get_feature_selection_events(limit: int = Query(100, ge=1, le=1000)):
    return {"events": feature_selection.list_events(limit=limit)}
