import os

import uvicorn
from api.errors.http_errors import http_error_handler, validation_exception_handler
from api.routers import index
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from routers import correction
from starlette.middleware.cors import CORSMiddleware


load_dotenv()
API_ADDRESS = os.getenv("API_ADDRESS")
API_PORT = int(os.getenv("API_PORT"))


def init_application() -> FastAPI:
    application = FastAPI(title="FTECH VietAC")
    origins = ["*"]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.add_exception_handler(HTTPException, http_error_handler)
    application.add_exception_handler(RequestValidationError, validation_exception_handler)

    application.include_router(router=index.router, tags=["Home"])
    application.include_router(router=correction.router, tags=["Spell Checking"])
    return application


if __name__ == "__main__":
    app = init_application()
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
