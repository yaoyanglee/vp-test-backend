from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio

from logger.Logger import Logger

sse_router = APIRouter()


@sse_router.get("/v1")
async def sse_v1():
    queue = asyncio.Queue()

    async def message_pusher():

        async def addToQueue(patientId: str, message: str):
            await queue.put((patientId, message))

        def updateQueue(patientId: str, message: str):
            asyncio.run(addToQueue(patientId, message))

        logger = Logger()
        logger.subscribe(updateQueue)

        while True:
            pid, msg = await queue.get()
            yield f"data: {pid}|||{msg}\n\n".encode()
            queue.task_done()

    return StreamingResponse(message_pusher(), media_type="text/event-stream")
