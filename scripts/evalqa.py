"""
Evaluate question-answer under sanitization + desanitization

Run like this:

python3 scripts/evalqa.py

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
from typing import List
from pydantic import BaseSettings
from dotenv import load_dotenv
import importlib
import re

from privacypromptrewriting.sanitize.sanitizer import sanitize_texts
from privacypromptrewriting.sanitize.desanitizer import desanitize_texts
from privacypromptrewriting.sanitize.qa import answer_questions
from privacypromptrewriting.sanitize.compare_answers import compare_answers
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

console = Console()
app = typer.Typer()

setup_colored_logging()


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


MONEY_NOISE = True

load_dotenv()

@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    examples: str = typer.Option(
        "binary_comparisons_far",
        "--examples",
        "-e",
        help="examples file under privacypromptrewriting/sanitize/data",
    ),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    output: str = typer.Option("evalqa.txt", "--output", "-o", help="output file"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )

    module = importlib.import_module(
        f"privacypromptrewriting.sanitize.data.{examples}"
    )
    samples = getattr(module, "examples")
    context_questions = [
        f"""
        {ctx}
        
        {q}
        """
        for ctx,q,ans in samples
    ]

    san_context_question_strs = sanitize_texts(context_questions)

    # split each san_context_question into context and question by
    # splitting on 2 or more newlines
    san_context_question_pairs = [
        split_paragraphs(cq)
        for cq in san_context_question_strs
    ]

    # (idx, grp, san-ctx, san-q, orig-c, orig-q, orig-a)
    idx_grp_sc_sq_c_q_a = [
        (i, i, cq[0], cq[1], samples[i][0], samples[i][1], samples[i][2]) for i, cq in
        enumerate(san_context_question_pairs) if len(cq) == 2
    ]

    grp_san_context_question_pairs = [
        (grp, sc,sq) for i,grp,sc,sq,c,q,a in idx_grp_sc_sq_c_q_a
    ]
    san_context_question_pairs = [
        (sc,sq) for g,sc,sq in grp_san_context_question_pairs
    ]
    groups = [g for g,sc,sq in grp_san_context_question_pairs]

    san_answers = answer_questions(san_context_question_pairs, llm=model)
    assert len(san_answers) == len(grp_san_context_question_pairs)

    # attach group number to each san_answers as a dict
    group_san_answers = [
        {"group": g, "answer": a} for g,a in zip(groups, san_answers)
    ]

    desan_answers = desanitize_texts(group_san_answers, money_noise=MONEY_NOISE)
    context_q_a_dsa = [
        (c,q,a,dsa) for (i,_,sc,sq,c,q,a), dsa in
        zip(idx_grp_sc_sq_c_q_a,  desan_answers)
    ]
    evals = compare_answers(context_q_a_dsa, llm="")
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

    print(f"N = {n}, {n_correct} correct ({metric:.2f}%), {problems} problems")

    # DUMP ALL RESULTS TO FILE
    all_stages = [
        (i, grp, c,q,a,sc,sq,sa,dsa,e)
        for (i, grp, sc, sq, c, q, a), sa, dsa, e in
        zip(idx_grp_sc_sq_c_q_a, san_answers, desan_answers, eval_nums)
    ]

    # dump all_stages to file
    # separated by ----
    with open(output, 'w') as file:
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




if __name__ == "__main__":
    app()
