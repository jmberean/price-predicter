from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import forecast_endpoint
from .api_schemas import ForecastRequest, ForecastResponse
from .auth import api_key_dependency
from .pipeline.forecast import ForecastEngine


def create_app(
    engine: ForecastEngine | None = None,
    api_key: str | None = None,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    app = FastAPI(title="Aetheris Oracle", version="0.1.0")
    eng = engine or ForecastEngine()
    app.state.api_key = api_key
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/forecast", response_model=ForecastResponse, dependencies=[Depends(api_key_dependency)])
    def forecast(payload: ForecastRequest) -> dict:
        try:
            return forecast_endpoint(payload.model_dump(), engine=eng)
        except Exception as exc:  # pragma: no cover - FastAPI will wrap
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
