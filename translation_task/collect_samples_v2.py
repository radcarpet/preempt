import re
import random_name_generator as rng
import math
import numpy as np
from pyfpe_ff3 import FF3Cipher, format_align_digits

import sys
sys.path.insert(0, '../universal-ner')
from src.utils import *
from utils_backup import *
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
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Translation eval for Preempt")
    parser.add_argument(
        "--task",
        type=str,
        default='de-en',
        help="de-en, fr-en, cs-en etc."
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
        type=str,
        default='cuda:0',
        help="Device to mount",
    )

    parser.add_argument(
        "--translation_model",
        type=str,
        default='gpt-4',
        help="Model for translation. Use path",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Crypto seed"
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
        default=None,
        help="Tag for saving results"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Samples for testing"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for online translation"
    )
    
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_fn(dataset, big_fp):
    with open(big_fp, 'w') as fp:
        json.dump(dataset, fp, indent=2, sort_keys=True)

def load_data(big_fp):
    with open(big_fp, 'r') as fp:
          data = json.load(fp)
    return data        

seed = 0
seed_everything(seed)

args = parse_args()
print(args)

model_path = args.ner_path

trans_model = args.translation_model
trans_model_paths_src2tar = {
    'opus-mt-fr-en': '/PATH/TO/models/opus-mt-en-fr',
    'opus-mt-de-en': '/PATH/TO/models/opus-mt-en-de',
}
trans_model_paths_tar2src = {
    'opus-mt-fr-en': '/PATH/TO/models/opus-mt-fr-en',
    'opus-mt-de-en': '/PATH/TO/models/opus-mt-de-en',
}


device = args.device
named_entity = args.entity
task = args.task
batch_size = args.batch_size
api_key = args.api_key

target_lang, source_lang = task.split('-')
lang_mapping = {
    'de': 'German',
    'fr': 'French',
}
if named_entity=='All':
    named_entities = ['Name','Age','Money',]# 'Zipcode']

else:
    named_entities = [named_entity]
    
max_new_tokens = 1024

# Data
data = load_dataset("wmt14", 
                    task, split='train',
                    # cache_dir="/PATH/TO/data/cache"
                   )
num_data_idx = []
for i in tqdm(range(len(data))[:10000]):
    en_sentence = data[i]['translation'][source_lang]
    if named_entity!='Name' and named_entity!='Full Name':
        if bool(re.search(r'\d', en_sentence)):num_data_idx.append(i)
    else:
        num_data_idx.append(i)

print(len(num_data_idx))
num_data_en = [data[i]['translation'][source_lang] for i in num_data_idx]
num_data_de = [data[i]['translation'][target_lang] for i in num_data_idx]

num_data_de = [y for _, y in sorted(zip(num_data_en, num_data_de), key=lambda pair: len(pair[0]))]
num_data_en = sorted(num_data_en, key=lambda item: len(item))

temp_en = []
temp_de = []
for i, text in tqdm(enumerate(num_data_en)):
    if len(text) >= 100:
        if len(temp_en) < 10000:
            temp_en.append(text)
            temp_de.append(num_data_de[i])
        else:
            break

num_data_en = temp_en
num_data_de = temp_de

valid_test_cases = []

for entity in named_entities:
    extracted = extract_entities_LLM(
                    num_data_en,
                    model_path,
                    batch_size=batch_size,
                    entity_type=entity,
                    device=device,
                )[entity]


    for i, vals in enumerate(extracted):
        if entity=='Name':
            if len(vals[0])>0:
                # print(vals)
                valid_test_cases.append(i)
        else:
            if len(vals)>0:
                # print(vals)
                valid_test_cases.append(i)
        # if len(valid_test_cases)==50:
        #     break
    
    print(f"Found {len(valid_test_cases)} test cases")

    
    en_entity_specific_data = [num_data_en[i] for i in valid_test_cases]
    de_entity_specific_data = [num_data_de[i] for i in valid_test_cases]


    target_lang, source_lang = task.split('-')
    temp_data = {
        f'{source_lang}_data': en_entity_specific_data,
        f'{target_lang}_data': de_entity_specific_data,
        'split': 'test',
    }

    with open(f'cache/{source_lang}_{target_lang}_data_{named_entity}_{len(valid_test_cases)}.json', 'w') as fp:
        json.dump(temp_data, fp, sort_keys=True, indent=4)

    print(f"Done with collecting for {named_entity}!")