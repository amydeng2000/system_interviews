import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from typing import Any, Dict
import random


class GPU:
    """
    A mock GPU class for creating mock inference functions
    Comments contain actual implementations if we were to run this on multiple GPUs
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        print(f"Mock initializing GPT-2 model on GPU {gpu_id}")
        # self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def run_inference(self, input_text: str) -> str:
        processing_time = random.uniform(0.5, 2)
        time.sleep(processing_time)
        print(f"Result from GPU {self.gpu_id}: {input_text.upper()}")
        return f"Result from GPU {self.gpu_id}: {input_text.upper()}"
        # inputs = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        # outputs = self.model.generate(inputs, max_length=500)
        # return self.tokenizer.decode(outputs[0])


class MessageQueue:
    def __init__(self, num_gpus: int):
        self.request_queue = asyncio.Queue()
        self.results: Dict[int, asyncio.Future] = {}
        self.shutdown_flag = asyncio.Event()
        self.gpus = [GPU(i) for i in range(num_gpus)]
        self.consumer_tasks = (
            []
        )  # holds references to the asynchronous tasks created for each GPU consumer

    async def put_request(self, item: Dict[str, Any]) -> None:
        """
        We create an incomplete future for the id in self.results, which is set to be the result when inference finishes
        """
        future = asyncio.Future()
        self.results[item["id"]] = future
        await self.request_queue.put(item)

    async def get_result(self, request_id: int) -> Any:
        """
        `await future` will always wait until the future is completed and set in _consumer_worker function by a GPU
        """
        future = self.results[request_id]
        result = await future
        del self.results[request_id]
        return result

    async def start_consumers(self) -> None:
        """
        `asyncio.create_task` creates tasks that runs concurrently in the event loop until they are finished or cancelled
        We keep track of them in self.consumer_tasks to make sure we await to finish them before shutting down the Message Queue
        """
        for gpu in self.gpus:
            task = asyncio.create_task(self._consumer_worker(gpu))
            self.consumer_tasks.append(task)

    async def _consumer_worker(self, gpu: GPU) -> None:
        """
        Each consumer worker waits for an item from the queue, then runs inference on a separate thread
        """
        while not self.shutdown_flag.is_set():
            try:
                item = await self.request_queue.get()
                # Run GPU inference in a separate thread to not block the event loop
                result = await asyncio.to_thread(gpu.run_inference, item["text"])
                self.results[item["id"]].set_result(result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in GPU {gpu.gpu_id}: {str(e)}")
            finally:
                self.request_queue.task_done()

    async def shutdown(self) -> None:
        self.shutdown_flag.set()
        for task in self.consumer_tasks:
            task.cancel()
        # await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        self.consumer_tasks.clear()


app = FastAPI()
message_queue = MessageQueue(num_gpus=4)


class InferenceRequest(BaseModel):
    text: str


class InferenceResponse(BaseModel):
    result: str


@app.on_event("startup")
async def startup_event():
    await message_queue.start_consumers()


@app.on_event("shutdown")
async def shutdown_event():
    await message_queue.shutdown()


@app.post("/inference/", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    request_id = id(request)
    await message_queue.put_request({"id": request_id, "text": request.text})
    try:
        result = await asyncio.wait_for(
            message_queue.get_result(request_id), timeout=10.0
        )
        return InferenceResponse(result=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timed out")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
