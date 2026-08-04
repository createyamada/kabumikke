import uvicorn
from fastapi import FastAPI
from routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware


def get_application():

    # インスタンスを生成
    app = FastAPI(title="kabumikke", version="1.0.0")

    # ミドルウェア設定
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api")

    @app.get("/")
    async def health_check():
        return {"status": "ok"}

    return app


app = get_application()
