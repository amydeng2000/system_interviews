import aiohttp
import asyncio
import time
from collections import Counter


async def send_inference_request(session, url, text, request_id):
    start_time = time.time()
    async with session.post(  # alternatively, we could create a new session here, but reusing is faster
        url, json={"text": f"{text} - Request {request_id}"}
    ) as response:
        result = await response.json()
        end_time = time.time()
        return request_id, result, end_time - start_time


async def main():
    url = "http://localhost:8000/inference/"
    num_requests = 10

    async with aiohttp.ClientSession() as session:  # this allows us to reuse connection for each http request
        tasks = []
        for i in range(num_requests):
            task = send_inference_request(session, url, f"Test inference {i}", i)
            tasks.append(task)

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    # Process results
    response_times = [r[2] for r in results]
    avg_response_time = sum(response_times) / len(response_times)
    max_response_time = max(response_times)
    min_response_time = min(response_times)

    # Count occurrences of each GPU
    gpu_counts = Counter([r[1]["result"].split(":")[0].split()[-1] for r in results])

    print(f"Total requests: {num_requests}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print(f"Max response time: {max_response_time:.2f} seconds")
    print(f"Min response time: {min_response_time:.2f} seconds")
    print("GPU usage distribution:")
    for gpu, count in gpu_counts.items():
        print(f"GPU {gpu}: {count} requests")

    # Print the first few results to verify correctness
    for i, (request_id, result, response_time) in enumerate(results[:10]):
        print(
            f"Request {request_id}: {result['result']} (Response time: {response_time:.2f}s)"
        )


if __name__ == "__main__":
    asyncio.run(main())
