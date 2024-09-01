"""
Run sanitizer agent.

Run like this:

python3 scripts/sanitizer.py

Use optional arguments to change the settings, e.g.:

-m "local" to use a model served locally at an OpenAI-API-compatible endpoint
[ Ensure the API endpoint url matches the one in the code below, or edit it. ]
OR
- m "litellm/ollama/llama2" to use any model supported by litellm
(see list here https://docs.litellm.ai/docs/providers)
[Note you must prepend "litellm/" to the model name required in the litellm docs,
e.g. "ollama/llama2" becomes "litellm/ollama/llama2",
"bedrock/anthropic.claude-instant-v1" becomes
"litellm/bedrock/anthropic.claude-instant-v1"]

-ns # no streaming
-d # debug mode
-nc # no cache
-ct momento # use momento cache (instead of redis)

For details on running with local Llama model, see:
https://langroid.github.io/langroid/blog/2023/09/14/using-langroid-with-local-llms/
"""
import typer
from rich.console import Console
from rich import print
from typing import List
from pydantic import BaseSettings
from dotenv import load_dotenv
import asyncio

from privacypromptrewriting.sanitize.sanitizer_agent import (
    SanitizerAgent,
    SanitizerAgentConfig,
)
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

console = Console()
app = typer.Typer()

setup_colored_logging()


MyLLMConfig = OpenAIGPTConfig.create(prefix="myllm")
my_llm_config = MyLLMConfig(
    chat_model="litellm/ollama/llama2",
    # or, other possibilities for example:
    # "litellm/bedrock/anthropic.claude-instant-v1"
    # "litellm/ollama/llama2"
    # "local/localhost:8000/v1"
    # "local/localhost:8000"
    stream=False,
    chat_context_length=2048,  # adjust based on model
)

def read_file_by_paragraphs(filename:str) -> List[str]:
    with open(filename, 'r') as file:
        content = file.read()
        # Splitting by two or more newline characters
        paragraphs = [p for p in content.split('\n\n') if p]
    return paragraphs

class CLIOptions(BaseSettings):
    model: str = ""

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to the privacy sanitizer!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    # use the appropriate config instance depending on model name
    if opts.model.startswith("litellm/") or opts.model.startswith("local/"):
        # e.g. litellm/ollama/llama2 or litellm/bedrock/anthropic.claude-instant-v1
        llm_config = my_llm_config
        llm_config.chat_model = opts.model

    else:
        llm_config = OpenAIGPTConfig(stream=False)

    config = SanitizerAgentConfig(llm=llm_config)

    async def _do_task(text: str, i:int) -> str:
        agent = SanitizerAgent(config)
        task = Task(
            agent,
            name=f"Sanitizer-{i}",
            default_human_response="",  # eliminate human response
            only_user_quits_root=False, # allow agent sanitizer method to quit via "DONE"
            llm_delegate=False,
            single_round=False,
        )
        sanitized = await task.run_async(text)
        return sanitized.content


    texts = read_file_by_paragraphs('datasets/q-a/finance.md')
    async def _do_all():
        return await asyncio.gather(
            *(_do_task(text, i) for i, text in enumerate(texts))
        )
    # show rich console spinner

    with console.status("[bold green]Sanitizing..."):
        sanitized_texts = asyncio.run(_do_all())

    # write out the sanitized texts to same file name with -sanitized appended
    with open('datasets/q-a/finance-sanitized.md', 'w') as file:
        file.write('\n\n'.join(sanitized_texts))

    print('Done!')


@app.command()
def main(
        debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
        model: str = typer.Option("", "--model", "-m", help="model name"),
        no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
        nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
        cache_type: str = typer.Option(
            "redis", "--cachetype", "-ct", help="redis or momento"
        ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    opts = CLIOptions(model=model)
    chat(opts)


if __name__ == "__main__":
    app()
