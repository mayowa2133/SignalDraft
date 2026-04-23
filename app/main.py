from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.container import container
from app.utils.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    container.close()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Local-first AI inbox triage and reply drafting agent for job seekers.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(router)
