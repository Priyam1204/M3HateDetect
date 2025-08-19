
import os
import time
import logging  
import asyncio

from dotenv import load_dotenv
from enum import Enum
from typing import List, Dict, Protocol, Optional, Tuple, Union

from openai import OpenAI, AsyncOpenAI, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage

load_dotenv()

# Allowed models
class AllowedModelId(str, Enum):
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    DEEPSEEK_R1 = "deepseek/deepseek-r1"
    DEEPSEEK_V3 = "deepseek/deepseek-v3"
    GPT_4_1 = "openai/gpt-4.1"
    CLAUDE_3_7_SONNET = "anthropic/claude-3.7-sonnet"
    O3_MINI = "openai/o3-mini"
    GPT_4_1_NANO = "openai/gpt-4.1-nano"
    GPT_4_1_MINI = "openai/gpt-4.1-mini"

# Client interface
class LLMClient(Protocol):
    def create_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        ...


class LLMAsyncClient(Protocol):
    async def create_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        ...

# --- Sync adapter ---
class OpenAIClientAdapter:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        key = api_key or os.environ.get("API_KEY")
        url = base_url or os.environ.get("BASE_URL")
        self._client = OpenAI(api_key=key, base_url=url)

    def create_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages
        )
        return resp.choices[0].message

# --- Async adapter ---
class AsyncOpenAIClientAdapter:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        key = api_key or os.environ.get("API_KEY")
        url = base_url or os.environ.get("BASE_URL")
        self._client = AsyncOpenAI(api_key=key, base_url=url)

    async def create_completion(
        self,
        model: str,
        messages: List[ChatCompletionMessageParam]
    ) -> ChatCompletionMessage:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages
        )
        return resp.choices[0].message

# Inference service with retries & logging
class LLMInferenceService:
    def __init__(
        self,
        sync_client: LLMClient,
        async_client: LLMAsyncClient,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.sync_client = sync_client
        self.async_client = async_client
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger("LLMInferenceService")

    # ----- Synchronous inference -----
    def get_responses(
        self,
        model_ids: List[AllowedModelId],
        messages: List[ChatCompletionMessageParam]
    ) -> Dict[AllowedModelId, ChatCompletionMessage]:
        """
        Returns a map of each model_id -> ChatCompletionMessage
        Retries on rate limits (429) with exponential backoff.
        """
        results: Dict[AllowedModelId, ChatCompletionMessage] = {}

        for model_id in model_ids:
            delay = self.backoff_factor
            for attempt in range(1, self.max_retries + 1):
                try:
                    self.logger.debug(f"[SYNC][{model_id.value}] attempt {attempt}")
                    msg = self.sync_client.create_completion(
                        model=model_id.value,
                        messages=messages
                    )
                    results[model_id] = msg
                    break
                except RateLimitError as e:
                    if attempt == self.max_retries:
                        self.logger.error(
                            f"Rate limit on {model_id.value} after {attempt} attempts: {e}"
                        )
                        raise
                    self.logger.warning(
                        f"429 on {model_id.value}, retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                except OpenAIError as e:
                    self.logger.error(f"API error for {model_id.value}: {e}")
                    raise
                except Exception:
                    self.logger.exception(f"Unexpected error on {model_id.value}")
                    raise

        return results

    # ----- Asynchronous inference -----
    async def get_responses_async(
        self,
        model_ids: List[AllowedModelId],
        messages: List[ChatCompletionMessageParam],
        concurrency: int = 5,
        fallback_to_sync: bool = True
    ) -> Dict[AllowedModelId, ChatCompletionMessage]:
        semaphore = asyncio.Semaphore(concurrency)
        results: Dict[AllowedModelId, ChatCompletionMessage] = {}

        async def _fetch(
            model_id: AllowedModelId
        ) -> Tuple[AllowedModelId, Union[ChatCompletionMessage, Exception]]:
            delay = self.backoff_factor
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with semaphore:
                        self.logger.debug(f"[ASYNC][{model_id.value}] attempt {attempt}")
                        msg = await self.async_client.create_completion(
                            model=model_id.value,
                            messages=messages
                        )
                        return model_id, msg
                except RateLimitError as e:
                    if attempt == self.max_retries:
                        self.logger.error(f"[ASYNC][{model_id.value}] rate limit: {e}")
                        return model_id, e
                    self.logger.warning(
                        f"[ASYNC][{model_id.value}] rate limited, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                except OpenAIError as e:
                    self.logger.error(f"[ASYNC][{model_id.value}] API error: {e}")
                    return model_id, e
                except Exception as e:
                    self.logger.exception(f"[ASYNC][{model_id.value}] unexpected: {e}")
                    return model_id, e

        # launch and gather all tasks
        tasks = [ _fetch(mid) for mid in model_ids ]
        completed = await asyncio.gather(*tasks)

        # process results + optional sync fallback
        for model_id, payload in completed:
            if isinstance(payload, Exception):
                self.logger.error(f"[ASYNC][{model_id.value}] error: {payload}")
                if fallback_to_sync:
                    self.logger.info(f"[ASYNC] falling back to sync for {model_id.value}")
                    try:
                        sync_res = self.get_responses([model_id], messages)
                        results[model_id] = sync_res[model_id]
                    except Exception as e:
                        self.logger.error(f"[ASYNC→SYNC][{model_id.value}] failed: {e}")
                # else: drop it
            else:
                results[model_id] = payload

        return results

# --- Example usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("LLMInferencePerf")

    sync_adapter  = OpenAIClientAdapter()
    async_adapter = AsyncOpenAIClientAdapter()
    service       = LLMInferenceService(sync_adapter, async_adapter, logger=logger)

    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Bitte schreibe einen kurzen Absatz auf Deutsch."}
    ]

    # --- Sync run over ALL models ---
    all_models = list(AllowedModelId)
    start_sync = time.perf_counter()
    sync_results = service.get_responses(all_models, messages)
    end_sync = time.perf_counter()
    logger.info(f"SYNC: ran {len(all_models)} models in {end_sync - start_sync:.2f}s")

    # print one sample to verify
    for mid, msg in sync_results.items():
        logger.info(f"[SYNC RESULT] {mid.value}: {msg.content[:60]!r}...")

    # --- Async run over ALL models ---
    async def timed_async_run():
        start_async = time.perf_counter()
        async_results = await service.get_responses_async(
            all_models, messages, concurrency=5, fallback_to_sync=True
        )
        end_async = time.perf_counter()
        logger.info(f"ASYNC: ran {len(all_models)} models in {end_async - start_async:.2f}s")
        for mid, msg in async_results.items():
            logger.info(f"[ASYNC RESULT] {mid.value}: {msg.content[:60]!r}...")
    asyncio.run(timed_async_run())