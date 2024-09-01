"""
RAG Question-answering for a list of:
[ (Passage, question, question, question),
  (Passage, question, question, question),
  ...
]
"""

from rich.console import Console
from typing import List, Tuple
from dotenv import load_dotenv
import asyncio

from langroid.agent.batch import llm_response_batch
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.logging import setup_colored_logging
from langroid.utils.constants import NO_ANSWER

console = Console()

setup_colored_logging()

load_dotenv()

def answer_questions(
        context_questions: List[Tuple[str,str]],
        llm: str
) -> List[str]:

    # use the appropriate config instance depending on model name
    if llm.startswith("litellm/") or llm.startswith("local/"):
        # e.g. litellm/ollama/llama2 or litellm/bedrock/anthropic.claude-instant-v1
        llm_config = OpenAIGPTConfig(
            chat_model=llm,
            stream=False,
            chat_context_length=4096,
        )
    else:
        llm_config = OpenAIGPTConfig(stream=False)

    config = ChatAgentConfig(
        llm=llm_config,
        system_message="""
        You are a helpful assistant.
        The user will send you a CONTEXT, followed by a QUESTION.
        You must answer the question concisely in no more than 3 sentences.
        If the question contains a numbered choice, ONLY indicate your choice as a 
        single number, and say NOTHING ELSE. 
        """
    )

    agent = ChatAgent(config)

    input_map = (
        lambda c_q:
        f"""
            CONTEXT:
            {c_q[0]}
            
            QUESTION: {c_q[1]}
            """
    )

    output_map = lambda answer: answer.content

    answers = llm_response_batch(
        agent,
        context_questions,
        input_map=input_map,
        output_map=output_map,
    )

    return answers

