import re
import random_name_generator as rng
import math
import numpy as np
from pyfpe_ff3 import FF3Cipher, format_align_digits

import sys
sys.path.insert(0, '../universal-ner')
from src.utils import *
from utils_backup import *
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import evaluate
import gc
import openai

from tqdm.auto import tqdm
import asyncio
import ast
import argparse
import json
import traceback 
import time
import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Translation eval for Preempt")
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Name, Money, Age, Zipcode, All"
    )
    
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Name, Money, Age, Zipcode, All"
    )
    
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="english, german, french"
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Device to mount",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path for NER comparison",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch processing"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="seed"
    )
    parser.add_argument(
        "--ner_path",
        type=str,
        default=None,
        help="NER model path",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=datetime.datetime.now(),
        help="Tag for saving results"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Samples for testing"
    )
    
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

seed = args.seed
seed_everything(seed)

model_path = args.ner_path
tag = args.tag
data_path = args.data_path
samples = args.samples
assert model_path!=None
assert tag!=None
assert data_path!=None

device = args.device
named_entity = args.entity
batch_size = args.batch_size
api_key = args.api_key

max_new_tokens = 1024

langs = [args.lang]
if args.lang=='all':
    langs = ['english', 'german', 'french']
entities = [args.entity]
if args.entity=='all':
    entities = ['Name', 'Age', 'Money']

# Preprocess
if model_path[-1]=='/': model_path=model_path[:-1]

# Data
if os.path.isfile(f"datasets/ner_extract_test_inputs_pii_{samples}.json"):
    data_dict = load_json_dataset(f"datasets/ner_extract_test_inputs_pii_{samples}.json")
    answers = load_json_dataset(f"datasets/ner_extract_test_label_pii_{samples}.json")
else:
    data = load_json_dataset(data_path)
    # Let's get 500 samples per entity, for each language.
    data_dict = {lang: {entity: [] for entity in entities} for lang in langs}
    answers = {lang: {entity: [] for entity in entities} for lang in langs}

    for entity in entities:
        for lang in langs:
            data_dict[lang][entity] = []
            answers[lang][entity] = []
            temp, labels = find_entity(data, entity, lang, samples)
            data_dict[lang][entity].extend(temp)
            answers[lang][entity].extend(labels)

    # print(data_dict)
    save_json_list(data_dict, f"datasets/ner_extract_test_inputs_pii_{samples}.json")
    save_json_list(answers, f"datasets/ner_extract_test_label_pii_{samples}.json")

for entity in entities:
    for lang in langs:
        print(f"Entity: {entity}, Lang: {lang}")
        
        # Uni-NER
        if 'uniner' in model_path.lower():
            save_path = f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json"
            if not os.path.isfile(save_path):
                extracted = extract_entities_LLM(
                    data_dict[lang][entity],
                    model_path,
                    batch_size=batch_size,
                    entity_type=entity,
                    device=device,
                )
                save_json_list(extracted, f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json")
            else:
                extracted = load_json_dataset(save_path)

            t_answers = ner_cleaner(answers, entity, lang)
            precision, recall, tp, fp, fn = ner_match_score_all(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', model_path.split('/')[-1])
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)
            print('\n')

        # Llama-3
        if 'llama' in model_path.lower():
            save_path = f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json"
            if not os.path.isfile(save_path):
                extracted = extract_entities_llama(
                    data_dict[lang][entity],
                    model_path,
                    batch_size=batch_size,
                    entity=entity,
                    device=device,
                )
                save_json_list(extracted, f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json")
            else:
                extracted = load_json_dataset(save_path)

            t_answers = ner_cleaner(answers, entity, lang)
            extracted = llama3_cleaner(extracted, entity)
            # print(t_answers, '\n')
            # print(extracted, '\n')
            precision, recall, tp, fp, fn = ner_match_score_all(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', model_path.split('/')[-1])
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)
            print('\n')
        
        # Gemma-2-9B-Inst
        if 'gemma' in model_path.lower():
            save_path = f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json"
            print(save_path)
            if not os.path.isfile(save_path):
                extracted = extract_entities_gemma(
                    data_dict[lang][entity],
                    model_path,
                    batch_size=batch_size,
                    entity=entity,
                    device=device,
                )
                save_json_list(extracted, f"results/{model_path.split('/')[-1]}_ner_extract_{entity}_{lang}.json")
            else:
                extracted = load_json_dataset(save_path)

            t_answers = ner_cleaner(answers, entity, lang)
            extracted = llama3_cleaner(extracted, entity)
            # print(t_answers, '\n')
            # print(extracted, '\n')
            precision, recall, tp, fp, fn = ner_match_score_all(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', model_path.split('/')[-1])
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)
            print('\n')
        
        # GPT-4
        if 'gpt' in model_path.lower():
            save_path = f"results/gpt-4-turbo-2024-04-09_ner_extract_{entity}_{lang}.json"
            if not os.path.isfile(save_path):
                extracted, _ = extract_entities_GPT(
                    data_dict[lang][entity],
                    entity=entity,
                    api_key = api_key,
                )
                save_json_list(extracted, f"results/gpt-4-turbo-2024-04-09_ner_extract_{entity}_{lang}.json")
            else:
                extracted = load_json_dataset(save_path)    

            t_answers = ner_cleaner(answers, entity, lang)
            extracted = gpt_cleaner(extracted, entity)
            precision, recall, tp, fp, fn = ner_match_score_all(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', 'GPT-4')
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)        
            print('\n')
        
        # Claude
        if 'claude' in model_path.lower():
            t_answers = ner_cleaner(answers, entity, lang)
    #         print("\nANSWERS\n",t_answers)
            extracted = load_claude_data(lang, entity)
    #         print("\nQUERIES\n",extracted)
            precision, recall, tp, fp, fn = ner_match_score_all(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', 'Claude Sonnet-3.5')
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)        
            print('\n')
        
        # Gemini
        if 'gemini' in model_path.lower():
            save_path = f"results/gemini_ner_extract_{entity}_{lang}.json"
            if not os.path.isfile(save_path):
                extracted, _ = extract_entities_gemini(
                    data_dict[lang][entity],
                    entity=entity,
                    lang=lang,
                    api_key = api_key,
                )
                save_json_list(extracted, f"results/gemini_ner_extract_{entity}_{lang}.json")
            else:
                extracted = load_json_dataset(save_path)    

            t_answers = ner_cleaner(answers, entity, lang)
    #         print("\nANSWERS\n",t_answers)
            extracted = gemini_cleaner(extracted, entity)
    #         print("\nQUERIES\n",extracted)
            precision, recall, tp, fp, fn = ner_match_score_gemini(extracted, t_answers, entity)
            f1_score = 2*precision*recall/(precision + recall)
            pprint('model', 'Gemini-1.5')
            pprint('entity', entity)
            pprint('lang', lang)
            pprint('precision', precision)
            pprint('recall', recall)
            pprint('f1-score', f1_score)        
        print("#"*30)
