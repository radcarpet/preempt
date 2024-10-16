# preempt
Code submission for USENIX-Security 2025

## Setup
Install libraries in a Python3.11 venv with `packagelist.txt`.

## Running Experiments
### Translation and NER

All relevant translation experiment code and details can be found under `translation_task`.

Using the [Universal-NER](https://github.com/universal-ner/universal-ner) models requires its own setup. Please follow instructions in the `universal-ner` folder.

### Medical QA

The code used for the medical QA experiments is available under `notebooks/medical_qa.ipynb`. 
- The data used for medical QA experiments is available under `datasets/medical-qa`. New data can be generated using the notebook.

### Financial QA

The code used for Financial QA experiments is available under `notebooks/financial_qa.ipynb`.

### RAG
Spin up local LLM using:

```
ollama pull mistral:7b-instruct-v0.2-q8_0
```

Run experiments like this, from the root of the repo:

```
python3 scripts/evalqa-rag.py \
    -e qa-cc-zip-date.txt  -nc -o evals  \
    -m litellm/ollama_chat/mistral:7b-instruct-v0.2-q8_0 
    -st aes
```
The `-o` option is the output dir (relative to root of repo) for results.

The last option `-st` option is to specify "sanitization type" (default is "fpe" if omitted).
 - `aes` = AES
 - `rand` = RAND
 - `fpe` = FPE
