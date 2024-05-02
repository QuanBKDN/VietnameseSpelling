import json
import time
from typing import Literal

import redis.asyncio as redis
from api.schema.item import RequestItem


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class RedisConnector(redis.Redis):
    __metaclass__ = SingletonMeta

    def __init__(
        self,
        host: str = "103.119.132.170",
        port: int = 6367,
        decode_responses: Literal[True, False] = True,
    ):
        super().__init__(host=host, port=port, decode_responses=decode_responses)

    async def push_request(self, request_item: RequestItem):
        """
        Push request to redis
        Args:
            request_item: RequestItem

        Returns: Request item with current id in redis

        """
        await self.incr("request_counter")
        current_id = await self.get("request_counter")
        request_item = {
            "id": current_id,
            "content": request_item.content,
            "prediction": None,
            "finish": int(False),
        }
        encode_data = json.dumps(request_item, indent=2).encode("utf-8")
        if await self.hset(name="request", key=f"rq_{current_id}", value=encode_data):
            return request_item

    async def get_response(self, request_id):
        """
        Check request status continuously, get it if prediction is ready
        Args:
            request_id: Request id need to be checked

        Returns: Return response if inference process have done

        """
        while True:
            response = await self.hgetall(name="request")
            response = json.loads(response[f"rq_{request_id}"])
            if response["prediction"] is not None:
                return response
            else:
                time.sleep(0.1)

    async def submit(self, request_id):
        """
        Delete a record in redis by request_id if response have been returned.
        Args:
            request_id: The id of request need to be deleted

        """
        await self.hdel("request", f"rq_{request_id}")

    async def get_queued_requests(self):
        """
        Get all request in queue

        Returns: List of requestItem

        """
        data = await self.hgetall(name="request")
        for key in data.keys():
            data[key] = json.loads(data[key])
        request_queue = [item[1] for item in data.items() if item[1]["prediction"] is None]
        request_queue = sorted(request_queue, key=lambda x: int(x["id"]))
        return request_queue

    async def clear_all(self):
        """
        Remove all record in redis
        """
        for key in await self.scan_iter("*"):
            await self.delete(key)

    async def push_response(self, responses):
        """
        Push all prediction to redis after inference
        Args:
            responses: Predictions from model
        """
        for item in responses:
            current_id = item["id"]
            encode_data = json.dumps(item, indent=2).encode("utf-8")
            await self.hset(name="request", key=f"rq_{current_id}", value=encode_data)

    async def get_all_item(self):
        """
        Print all request in redis
        """
        print(await self.hgetall("request"))
