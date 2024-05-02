import fastapi


router = fastapi.APIRouter()


@router.get("/")
async def index():
    return {"message": "Use /docs for more detail about API!"}
