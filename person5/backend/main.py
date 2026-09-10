from fastapi import FastAPI

from .analysis import router as analysis_router
from .auth import router as auth_router
from .upload import router as upload_router


app = FastAPI(title="SatQuery API")
app.include_router(auth_router)
app.include_router(upload_router)
app.include_router(analysis_router)
