import asyncio
import os
import time
from multiprocessing import Process

import torch
from api.database.redis_connector import RedisConnector
from dotenv import load_dotenv
from vietac import Corrector


load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH")
DICTIONARY_PATH = os.getenv("DICTIONARY_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceProcess:
    def __init__(self, model_path=MODEL_PATH, dictionary_path=DICTIONARY_PATH, device=device):
        self.redis_connector = RedisConnector()
        self.model_path = model_path
        self.dictionary_path = dictionary_path
        self.device = device

    async def start(self, main_process: bool = True):
        """
        Start Inference process, waiting for request.
        Args:
            main_process: (`bool`) Start inference in main process or subprocess

        """
        if main_process:
            await self.wait()
        else:
            Process(target=self.wait).start()

    async def wait(self):
        """
        Wait for request
        """
        corrector = Corrector(
            model_path=self.model_path, dictionary_path=self.dictionary_path, device=self.device
        )
        while True:
            response = await self.infer(corrector)
            if not response:
                time.sleep(0.1)
            else:
                await self.redis_connector.push_response(response)

    async def infer(self, corrector: Corrector, batch_size: int = 16):
        """
        Get request from redis then pass into corrector
        Args:
            corrector: Infer model
            batch_size: Maximum record can be passed into model

        Returns: Prediction by model

        """
        queued_request = await self.redis_connector.get_queued_requests()
        if len(queued_request) == 0:
            return False
        queued_request = queued_request[:batch_size]
        text_list = [item["content"] for item in queued_request][:batch_size]
        responses = corrector.infer(text_list)
        for i, response in enumerate(responses):
            queued_request[i]["prediction"] = response
        return queued_request


if __name__ == "__main__":
    inference_process = InferenceProcess()
    asyncio.run(inference_process.start(main_process=True))
