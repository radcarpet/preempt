# preempt
Code submission for USENIX-Security

## Setup
Install libraries in a conda environment with `packagelist.txt` and `requirements.txt`.

## Running Experiments
### Translation and NER

All relevant translation experiment code and details can be found under `translation_task`.

Using the [Universal-NER](https://github.com/universal-ner/universal-ner) models requires its own setup. Please follow instructions in the `universal-ner` folder.

### Medical QA

The code used for the medical QA experiments is available under `notebooks/medical_qa.ipynb`. 
- The data used for medical QA experiments is available under `datasets/medical-qa`. New data can be generated using the notebook.

### Financial QA

The code used for Financial QA experiments is available under `notebooks/financial_qa.ipynb`.
