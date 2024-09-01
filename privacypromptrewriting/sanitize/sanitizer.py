from rich.console import Console
from typing import List
from dotenv import load_dotenv
import asyncio

from privacypromptrewriting.sanitize.sanitizer_agent import (
    SanitizerAgent,
    SanitizerAgentConfig,
)
import json
from langroid.agent.task import Task
from langroid.agent.batch import run_batch_tasks
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.logging import setup_colored_logging

console = Console()

setup_colored_logging()

load_dotenv()

def sanitize_texts(texts: List[str], llm: str="", money_noise:bool=True) -> List[str]:
    # e.g. litellm/ollama/llama2 or litellm/bedrock/anthropic.claude-instant-v1
    llm_config = OpenAIGPTConfig(
        chat_model=llm,
        stream=False,
        chat_context_length=4096,
    )

    config = SanitizerAgentConfig(llm=llm_config, money_noise=money_noise)
    agent = SanitizerAgent(config)
    task = Task(
        agent,
        name="Sanitizer",
        default_human_response="",  # eliminate human response
        only_user_quits_root=False, # allow agent sanitizer method to quit via "DONE"
        llm_delegate=False,
        single_round=False,
    )

    # make a list of json objects with content and number
    numbered_texts = [
        {
            "content": text,
            "number": i
        }
        for i, text in enumerate(texts)
    ]
    # convert to json strings , one per item
    json_texts = [json.dumps(nt) for nt in numbered_texts]


    input_map = lambda text: text
    output_map = lambda sanitized: sanitized.content

    sanitized = run_batch_tasks(
        task,
        json_texts,
        input_map=input_map,
        output_map=output_map,
    )

    return sanitized

