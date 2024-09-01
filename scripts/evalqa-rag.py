"""
Evaluate RAG question-answer under sanitization + desanitization

Run like this:

python3 scripts/evalqa-rag.py

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
import pandas as pd
from rich.console import Console
from typing import List
from pydantic import BaseSettings, BaseModel
from dotenv import load_dotenv
import importlib
import os
from pathlib import Path
import re

import langroid.language_models as lm
from privacypromptrewriting.sanitize.globals import SanitizationState
from privacypromptrewriting.sanitize.sanitizer import sanitize_texts
from privacypromptrewriting.sanitize.desanitizer import desanitize_texts
from privacypromptrewriting.sanitize.qa import answer_questions
from privacypromptrewriting.sanitize.compare_answers import compare_answers
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from pyparsing import *

console = Console()
app = typer.Typer()

setup_colored_logging()

class QAPair(BaseModel):
    question: str
    answer: str

class RagQA(BaseModel):
    passage: str
    qa_pairs: List[QAPair]

class ContextQA(BaseModel):
    passage: str
    question: str
    answer: str

# Define the parsing rules
P = Literal("P:").suppress()
Q = Literal("Q:").suppress()
A = Literal("A:").suppress()

# Define TEXT to capture multiple lines until Q: is encountered
TEXT = originalTextFor(OneOrMore(~Q + restOfLine + LineEnd(), stopOn=Q | StringEnd()))


question = Q + SkipTo(A)("question")
answer = A + SkipTo(Q | P | StringEnd())("answer")
qa_pair = Group(question("question") + answer("answer"))
passage = Group(P + TEXT("passage") + OneOrMore(qa_pair)("qa_pairs"))
grammar = OneOrMore(passage)

def parse_ragqa(text: str) -> List[ContextQA]:
    result = grammar.parseString(text, parseAll=True).asList()

    rag_qas = [
        RagQA(
            passage=x[0],
            qa_pairs=[
                QAPair(question=qa[0], answer=qa[1])
                for qa in x[1:]
            ]
        ) for x in result
    ]
    # convert to list of ContextQA
    # rag_qas = [
    #     ContextQA(
    #         passage=rag_qa.passage,
    #         question=qa.question,
    #         answer=qa.answer
    #     ) for rag_qa in rag_qas for qa in rag_qa.qa_pairs
    # ]
    return rag_qas


def parse_ragqa_file(filename: str) -> List[ContextQA]:
    with open(filename, 'r') as file:
        content = file.read()
        return parse_ragqa(content)

def split_paragraphs(text: str) -> List[str]:
    """
    Split the input text into paragraphs using "\n\n" as the delimiter.

    Args:
        text (str): The input text.

    Returns:
        list: A list of paragraphs.
    """
    # Split based on a newline, followed by spaces/tabs, then another newline.
    paras = re.split(r"\n[ \t]*\n", text)
    return [para.strip() for para in paras if para.strip()]

def read_file_by_paragraphs(filename:str) -> List[str]:
    with open(filename, 'r') as file:
        content = file.read()
        # Splitting by two or more newline characters
        paragraphs = [p for p in content.split('\n\n') if p]
    return paragraphs

load_dotenv()

@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    examples: str = typer.Option(
        "ecommerce-customer-qa-email-phone.txt",
        "--examples",
        "-e",
        help="examples txt file under privacypromptrewriting/sanitize/data",
    ),
    sanitize_type : str = typer.Option("fpe", "--sanitize_type", "-st",
                                       help="sanitization type"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    output_dir: str = typer.Option("evalqa.txt", "--output", "-o", help="output dir"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    money_fpe: bool = typer.Option(False, "--money_fpe", "-mf",
                                   help="do FPE for money?"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    globals = SanitizationState.get_instance()
    globals.sanitize_type = sanitize_type
    model = model or lm.OpenAIChatModel.GPT4_TURBO
    data_dir = "privacypromptrewriting/sanitize/data"
    samples = parse_ragqa_file(Path(data_dir) / examples)

    newlines2 = "\n\n"
    # context_questions = [
    #     f"""
    #     {s.passage}
    #
    #     {newlines2.join(qa.question for qa in s.qa_pairs)}
    #     """
    #     for s in samples
    # ]

    # ONLY sanitize the contexts since sensitive info is in the context ONLY
    san_contexts = sanitize_texts(
        [s.passage for s in samples], llm=model, money_noise=not money_fpe,
    )

    # now join each sanitized context with its questions
    san_context_question_strs = [
        f"""
        {sc}
        
        {newlines2.join(qa.question for qa in s.qa_pairs)}
        """
        for sc,s in zip(san_contexts, samples)
    ]

    # save these to a file san-context-questions.txt under data-dir
    with open(Path(data_dir) / "san-context-questions.txt", 'w') as file:
        file.write("\n\n".join(san_context_question_strs))

    print(
        f"""
        Sanitized contexts and questions written to "
        {data_dir}/san-context-questions.txt
        """
    )


    # split each san_context_question into context and series of questions
    # splitting on 2 or more newlines
    grp_sc_sq_c_q_a = [] # group, san_context, san_question, context, question, answer
    for i, s in enumerate(san_context_question_strs):
        passage_questions = split_paragraphs(s) # => passage, q1, q2, ...
        assert len(passage_questions) == len(samples[i].qa_pairs) + 1
        passage = passage_questions[0]
        questions = passage_questions[1:]
        grp_sc_sq_c_q_a.extend(
            [i, passage, q, samples[i].passage, qa.question, qa.answer]
            for q, qa in zip(questions, samples[i].qa_pairs)
        )
    idx_grp_sc_sq_c_q_a = [
        (i, grp, sc, sq, c, q, a)
        for i, (grp, sc, sq, c, q, a) in enumerate(grp_sc_sq_c_q_a)
    ]

    grp_san_context_question_pairs = [
        (grp, sc,sq) for i,grp,sc,sq,c,q,a in idx_grp_sc_sq_c_q_a
    ]
    san_context_question_pairs = [
        (sc,sq) for g,sc,sq in grp_san_context_question_pairs
    ]
    groups = [g for g,sc,sq in grp_san_context_question_pairs]

    # answer with gpt4
    san_answers = answer_questions(
        san_context_question_pairs,
        llm=lm.OpenAIChatModel.GPT4_TURBO
    )
    assert len(san_answers) == len(grp_san_context_question_pairs)

    # attach group number to each san_answers as a dict
    group_san_answers = [
        {"group": g, "answer": a} for g,a in zip(groups, san_answers)
    ]

    desan_answers = desanitize_texts(group_san_answers, money_noise=not money_fpe)
    context_q_a_dsa = [
        (c,q,a,dsa) for (i,_,sc,sq,c,q,a), dsa in
        zip(idx_grp_sc_sq_c_q_a,  desan_answers)
    ]
    # Use GPT4_TURBO to evaluate the answers - this is fine since it's just
    # replacing our (human) eval.
    evals = compare_answers(context_q_a_dsa, llm=lm.OpenAIChatModel.GPT4_TURBO)
    same_ans = [1*(a.strip() == dsa.strip())  for (c,q,a,dsa) in context_q_a_dsa]

    def str2int(x):
        try:
            return int(x)
        except ValueError:
            return -1
    # use best of LLM eval or "same_ans" indicator -- i.e. if same ans we
    # consider it correct
    eval_nums = [max(s, str2int(e))  for e,s in zip(evals, same_ans)]
    problems = sum(1 for e in eval_nums if e == -1)
    n_correct = sum(1 for e in eval_nums if e == 1)
    n = len(evals)

    metric = 100*n_correct/n

    stats_file = Path(output_dir) / "stats.txt"

    stats = f"N = {n}, {n_correct} correct ({metric:.2f}%), {problems} problems"
    # write stats to stats_file
    with open(stats_file, 'w') as file:
        file.write(stats)
    print(stats)

    # DUMP ALL RESULTS TO FILE
    all_stages = [
        (i, grp, c,q,a,sc,sq,sa,dsa,e)
        for (i, grp, sc, sq, c, q, a), sa, dsa, e in
        zip(idx_grp_sc_sq_c_q_a, san_answers, desan_answers, eval_nums)
    ]

    # gather all these into a list of dicts and make a pandas df
    # for easy analysis
    df = pd.DataFrame(
        [
            {
                "idx": i,
                "group": grp,
                "context": c,
                "question": q,
                "answer": a,
                "sanitized_context": sc,
                "sanitized_question": sq,
                "sanitized_answer": sa,
                "desanitized_answer": dsa,
                "evaluation": e
            }
            for (i, grp, c, q, a, sc, sq, sa, dsa, e) in all_stages
        ]
    )
    # save df to a file named similar to output but with csv extension
    # get the extension of output, then replace the part after . with csv
    # do not assume output has .txt
    csv_output = Path(output_dir) / "test.csv"
    df.to_csv(csv_output, index=False)
    print(f"Results table written to {csv_output}")



    # dump all_stages to file
    # separated by ----
    txt_output = Path(output_dir) / "test.txt"
    with open(txt_output, 'w') as file:
        for i,grp,c,q,a,sc,sq,sa,dsa,e in all_stages:
            file.write(f"""
            ----
            [{grp}.{i}] CONTEXT:
            {c}
            
            QUESTION: {q}
            
            ANSWER: {a}
            
            SANITIZED-CONTEXT:
            {sc}
            
            SANITIZED-QUESTION: {sq}
            
            SANITIZED-ANSWER: {sa}
            
            DESANITIZED-ANSWER: {dsa}
            
            EVALUATION: {e}
            """)

    print(f"Results written to {txt_output}")
if __name__ == "__main__":
    app()
