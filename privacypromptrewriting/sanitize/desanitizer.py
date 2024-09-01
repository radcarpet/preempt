from rich.console import Console
from typing import Any, Dict, List
from dotenv import load_dotenv
import asyncio
import json

from privacypromptrewriting.sanitize.desanitizer_agent import (
    DesanitizerAgent,
    DesanitizerAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.batch import run_batch_tasks
from langroid.utils.logging import setup_colored_logging

console = Console()

setup_colored_logging()

load_dotenv()

def desanitize_texts(texts: List[Dict[str,Any]], money_noise:bool=False) -> List[str]:
    # texts = list of dicts with fields "group", "answer"
    config = DesanitizerAgentConfig(money_noise=money_noise)
    agent = DesanitizerAgent(config)
    task = Task(
        agent,
        name=f"DeSanitizer",
        default_human_response="",  # eliminate human response
        only_user_quits_root=False, # allow agent sanitizer method to quit via "DONE"
        llm_delegate=False,
        single_round=True,
    )

    input_map = lambda text: json.dumps(text)
    output_map = lambda desanitized: desanitized.content

    desanitized = run_batch_tasks(
        task,
        texts,
        input_map=input_map,
        output_map=output_map,
    )

    return desanitized