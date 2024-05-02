import os
import time
from typing import Union

import fastapi
import torch

from api.schema.item import (
    BaseResponseItem,
    ExceptionResponseItem,
    RequestItem,
    BatchResponseItem,
    BatchRequestItem,
)
from dotenv import load_dotenv

from vietac import Corrector
from vietac.dataset.preprocess import prepare_inference_data

router = fastapi.APIRouter()
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
DICTIONARY_PATH = os.getenv("DICTIONARY_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

corrector = Corrector(model_path=MODEL_PATH, dictionary_path=DICTIONARY_PATH)


@router.post(
    "/api/v1/spell_checking", response_model=Union[BaseResponseItem, ExceptionResponseItem]
)
async def check(request: RequestItem) -> Union[BaseResponseItem, ExceptionResponseItem]:
    start_time = time.time()
    text = prepare_inference_data(request.content)
    result, suggestion = corrector.infer(text)
    exec_time = time.time() - start_time
    return BaseResponseItem(
        content=request.content,
        result=result[0],
        suggestion=suggestion,
        elapsed_time="{:.4f}".format(exec_time),
    )


@router.post(
    "/api/v1/spell_checking_batch", response_model=Union[BatchResponseItem, ExceptionResponseItem]
)
async def check_batch_v1(request: BatchRequestItem):
    start_time = time.time()
    text = prepare_inference_data(request.content)
    result, suggestion = corrector.infer(text)
    exec_time = time.time() - start_time
    return BatchResponseItem(
        content=request.content,
        result=result,
        suggestion=suggestion,
        elapsed_time="{:.4f}".format(exec_time)
    )


@router.post(
    "/api/v2/spell_checking", response_model=Union[BatchResponseItem, ExceptionResponseItem]
)
async def check_batch(request: BatchRequestItem):
    start_time = time.time()
    text = prepare_inference_data(request.content)
    result, suggestion = corrector.infer(text)
    exec_time = time.time() - start_time
    return BatchResponseItem(
        content=request.content,
        result=result,
        suggestion=suggestion,
        elapsed_time="{:.4f}".format(exec_time)
    )
