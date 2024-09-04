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

"""
# PIPELINE:

1. Use the finetuned NER model (Uni-NER)/other open source models to extract entities.
2. Substitute with the crypto scheme
3. Translate with offline/online models
4. Use the finetuned NER model/other open source models to extract entities
5. Substitute with the crpto scheme
6. Comparisons
"""


##### 0. Data preprocessing

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
        default=1,
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
        default=50,
        help="Samples for testing"
    )
    
    parser.add_argument(
        "--exact_indices",
        type=str,
        default=None,
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
print("Data preprocessing...")
num_data_en = []
num_data_de = []
for entity in named_entities:
    with open(f'cache/{source_lang}_{target_lang}_data_{entity}.json', 'r') as fp:
        data = json.load(fp)

    num_data_en.extend(data[f'{source_lang}_data'])
    num_data_de.extend(data[f'{target_lang}_data'])

# exact_indices = load_data(args.exact_indices)['matching_indices']
# num_data_en = [num_data_en[i] for i in exact_indices]
# num_data_de = [num_data_de[i] for i in exact_indices]
num_data_en = num_data_en[:args.samples]
num_data_de = num_data_de[:args.samples]

print('Data size:', len(num_data_en))
assert len(num_data_en[0]) >= 100

print("\nSAMPLE:")
print(num_data_en[-1])
print(num_data_de[-1])

# Sanitize data:

N = {'Money': 100000} 
rho = {'Money': .05} 
epsilon = 1
key = "EF4359D8D580AA4F7F036D6F04FC6A94"
tweak = "D8E7920AFA330A73" 
c = FF3Cipher(key, tweak,allow_small_domain=True, radix=10)

all_data = {
    'extraction': {},
    'data_raw': [],
    'data_encrypted': [],
    'data_encrypted_translated': [],
    'translated_data_decrypted': [],
    'data_plain_translated': [],
    'args': vars(args)
}

extracted_dict = {
    'src_lang': source_lang,
    'tar_lang': target_lang,
    'src_extraction': {},
    'tar_extraction': {},
    'src_encrypted': {},
    'tar_decrypted': {},
}

##### 1. NER extraction
tag = f'{args.tag}'
data = num_data_en
# print(data, '\n')
# if not os.path.exists(tag):
if True:
    os.makedirs(tag, exist_ok=True)
    
    print("\nNER extraction...")

    for entity in named_entities:
        if 'uniner' in model_path:
            extracted = extract_entities_LLM(
                            data,
                            model_path,
                            batch_size=batch_size,
                            entity_type=entity,
                            device=device,
                        )[entity]
            
        elif 'gemma' in model_path:
            extracted = extract_entities_gemma(
                data, model_path, batch_size, entity, device
            )[entity]
            
        else:
            extracted = extract_entities_llama(
                data, model_path, batch_size, entity, device
            )[entity]

        extracted_dict['src_extraction'][entity] = extracted

    print(extracted[-1], '\n')
    # Save this!
    all_data['extraction'] = extracted_dict
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)


    ##### 2. Sanitation
    print("Sanitation...")
    for entity in named_entities:
        encrypted_values = get_encrypted_values(
            inputs=extracted_dict['src_extraction'][entity],
            entity=entity,
            N=N, rho=rho, epsilon=epsilon, c=c,)

        extracted_dict['src_encrypted'][entity] = encrypted_values


    print(encrypted_values[-1], '\n')
    # Do entity based substitution. For a list of lists, return a list of lists.
    data_encrypted = []
    # Get the ith data string.
    for i, line in enumerate(data):
        # Substitute for all entities
        for entity in named_entities:
            # Get extracted/encrypted data for the ith line.
            # Substitute all values.
            # print(extracted_dict['src_extraction'][entity][i])
            # print(extracted_dict['src_encrypted'][entity][i])
            for value, encrypt in zip(extracted_dict['src_extraction'][entity][i], extracted_dict['src_encrypted'][entity][i]):
                # print(value, encrypt)
                line = line.replace(value, encrypt)
                # print(line)

        data_encrypted.append(line)

    # Save this!
    all_data['data_raw'] = data
    all_data['data_encrypted'] = data_encrypted
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_encrypted[-1], '\n')
    
    ##### 3. Translation
    print("Translation...")
    # Translate with GPT:
    if trans_model=='gpt-4':
        data_encrypted_translated = translate_GPT(data_encrypted, lang_mapping[target_lang], api_key)
    elif trans_model=='gemini':
        data_encrypted_translated = translate_gemini(data_encrypted, lang_mapping[target_lang], api_key)
    else:
        data_encrypted_translated = translate_OPUS_MT(data_encrypted, trans_model_paths_src2tar[trans_model])

    # Save this!
    all_data['data_encrypted_translated'] = data_encrypted_translated
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(data_encrypted_translated[-1], '\n')

    ##### 4. NER Extraction
    print("Extraction...")
    for entity in named_entities:
        if 'uniner' in model_path:
            extracted = extract_entities_LLM(
                            data_encrypted_translated,
                            model_path,
                            batch_size=batch_size,
                            entity_type=entity,
                            device=device,
                        )[entity]
        
        elif 'gemma' in model_path:
            extracted = extract_entities_gemma(
                data_encrypted_translated, model_path, batch_size, entity, device
            )[entity]
        
        else:
            extracted = extract_entities_llama(
                data_encrypted_translated, model_path, batch_size, entity, device
            )[entity]

        extracted_dict['tar_extraction'][entity] = extracted

    # Save this!
    all_data['extraction'] = extracted_dict
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)

    print(extracted[-1], '\n')

    ##### 5. Decryption
    print("Decrpytion...")
    # Do entity based substitution. For a list of lists, return a list of lists.
    data_decrypted = []
    # Get the ith data string.
    for i, line in enumerate(data_encrypted_translated):
        # Substitute for all entities
        for entity in named_entities:
            # Get extracted/encrypted data for the ith line.
            # Substitute all values.
            for value, decrypt in zip(extracted_dict['tar_extraction'][entity][i], extracted_dict['src_extraction'][entity][i]):
                line = line.replace(value, decrypt)

        data_decrypted.append(line)

    # Save this!
    all_data['translated_data_decrypted'] = data_decrypted
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)
    
    print(data_decrypted[-1], '\n')
    
    
    ##### 6. Plain translation, for metrics.
    print("Plain translation...")
    # Translate with GPT:
    if trans_model=='gpt-4':
        data_plain_translated = translate_GPT(data, lang_mapping[target_lang], api_key)
    elif trans_model=='gemini':
        data_plain_translated = translate_gemini(data, lang_mapping[target_lang], api_key)
    else:
        data_plain_translated = translate_OPUS_MT(data, trans_model_paths_src2tar[trans_model])

    # Save this!
    all_data['data_plain_translated'] = data_plain_translated
    tag = f'{args.tag}/all_data.json'
    save_fn(all_data, tag)
        
    print(data_plain_translated[-1], '\n')

else:
    all_data = load_data(f'{args.tag}/all_data.json')
    extracted_dict = all_data['extraction']    
    
##### 7. Metrics
print("Metrics...")
print("References:")
num_data_de = [[i] for i in num_data_de]
print(num_data_de[:5], '\n')
# Score translation:
metric = evaluate.load('bleu')
predictions = all_data['translated_data_decrypted']
print("Preempt Translations:")
print(predictions[:5], '\n')
references = num_data_de
bleu_score = metric.compute(predictions=predictions, references=references)
print(f"\nSanitized:\n", bleu_score)

all_data['sanitized_bleu_score'] = bleu_score

metric = evaluate.load('bleu')
predictions = all_data['data_plain_translated']
print("\nPlain Translations:")
print(predictions[:5], '\n')
references = num_data_de
bleu_score = metric.compute(predictions=predictions, references=references)
print(f"\nPlain:\n", bleu_score)

all_data['plain_bleu_score'] = bleu_score

tag = f'{args.tag}/all_data.json'
save_fn(all_data, tag)