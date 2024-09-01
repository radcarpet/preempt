"""
Question-answering for a list of (question, context) pairs
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

def compare_answers(
        context_question_answer1_answer2: List[Tuple[str,str,str,str]],
        llm: str
) -> List[str]:

    llm_config = OpenAIGPTConfig(
        chat_model=llm,
        stream=False,
        chat_context_length=4096,
    )

    config = ChatAgentConfig(
        llm=llm_config,
        system_message="""
        You are a helpful assistant, expert at discerning the difference between two answers
        to a given question.
        
        The user will send you a CONTEXT, a QUESTION, and TWO ANSWERS, 
        labeled ANSWER-1 and ANSWER-2.
        ANSWER-1 is a known correct answer, 
        but ANSWER-2 is a candidate alternative answer.
        
        You must decide whether ANSWER-2 is a good answer or not.
        
        Remember:
        (a) If ANSWER-2 is a good answer, say 1 else say 0, and SAY NOTHING ELSE.
        (b) If ANSWER-2 is SAME as ANSWER-1, consider ANSWER-2 to be a good answer,
            and say 1.
        """
    )

    agent = ChatAgent(config)
    input_map = (
        lambda c_q_a1_a2:
            f"""
            CONTEXT:
            {c_q_a1_a2[0]}
        
            QUESTION: {c_q_a1_a2[1]}
        
            ANSWER-1: {c_q_a1_a2[2]}
        
            ANSWER-2: {c_q_a1_a2[3]}
            """
    )

    output_map =  lambda result: result.content

    answers = llm_response_batch(
        agent,
        context_question_answer1_answer2,
        input_map=input_map,
        output_map=output_map,
    )

    return answers

