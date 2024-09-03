import re
import random_name_generator as rng
import names
import math
import numpy as np
from pyfpe_ff3 import FF3Cipher, format_align_digits

import sys
sys.path.insert(0, '../universal-ner')
from src.utils import *
from utils_backup import *
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, MarianMTModel
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

import random

import unicodedata

from transformers import logging
import os

from fastchat.conversation import get_conv_template


logging.set_verbosity_error()



### pretty print
def print_block(vals):
    for val in vals:
        pprint(val)
    print('#'*30)
    
def pprint(tag, val):
    if isinstance(val, str):
        print(f"{tag} {'.'*(30 - len(val) -len(tag))} {val}")
    else:
        print(f"{tag} {'.'*(30 - 5 -len(tag))} {val:.3f}")

### json dataloader
def load_json_dataset(data_path):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    return data

### json saver
def save_json_list(data, path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=2)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

### Dataloader for pipelining
class any2en(Dataset):
    def __init__(self, text):
        self.text = text
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

### Sanitation helper
def clean_suffix_name(all_text):
    outputs = all_text
    suffixes = ['Prof ', 'Prof.', 'Prof', 'Mrs. ', 'Mrs ', 'Mrs.', 'Mrs', 'Mr ', 'Mr. ', 'Mr.', 'Mr', 'Ms. ', 'Ms.', 'Ms', 'Mrs ', 'Mrs', 'Ms ', 'Ms', 'Herr ', 'Herrn ', 'Frau ', 'M. ', 'Mme. ', 'M ', 'Mme ', 'Madame ', 'Monsieur ', 'Monsieur ', '\"', '"', '[', ']', '\\ ', 'Dr. ', 'Dr.', 'Miss ', 'Miss']
    for suffix in suffixes:
        outputs = [output.replace(suffix, '') for output in outputs]
    outputs = ['"[' + output + ']"' for output in outputs]
    return outputs

def postprocess_output(outputs, output_dict, entity_type):
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_suffix_name(outputs) 
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = [output.replace(",", "") for output in outputs]
        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]
    for output in outputs:
        if len(output) > 0: 
            if entity_type=="Name" or entity_type=="Full Name":
                temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                temp = temp.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.split(", ")
                output_dict[entity_type].append(temp)
            elif entity_type=="Money":
                temp = ast.literal_eval(output)
                temp = [t.replace('-','') for t in temp]
                output_dict[entity_type].append(temp)
            else:
                output_dict[entity_type].append(ast.literal_eval(output))
                
    return output_dict

### NER helpers
def postprocess_output_ner(outputs, output_dict, entity_type):
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = [output.replace(",", "") for output in outputs]
        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]
    
    temp = []
    for output in outputs:
        if entity_type=="Age":
            output = ast.literal_eval(output)
            temp.extend(output)
        elif entity_type=="Money":
            temp = ast.literal_eval(output)
            temp = [t.replace('-','') for t in temp]
        else:
            temp.append(output)
    
    output_dict[entity_type].append(temp)
    return output_dict

def postprocess_output_gpt(outputs, output_dict, entity_type):
    if entity_type=="Name" or entity_type=="Full Name":
        outputs = clean_suffix_name(outputs) 
    if entity_type=="Age":
        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
    if entity_type=="Zipcode":
        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
    if entity_type=="Money":
        outputs = [output.replace(",", "") for output in outputs]
        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]
    
    temp = []
    for output in outputs:
        if len(output) > 0: 
            if entity_type=="Name" or entity_type=="Full Name":
                _temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                _temp = _temp.replace('[', '')
                _temp = _temp.replace(']', '')
                _temp = _temp.split(", ")
                temp.extend(_temp)
            elif entity_type=="Age":
                output = ast.literal_eval(output)
                temp.extend(output)
            elif entity_type=="Money":
                temp = ast.literal_eval(output)
                temp = [t.replace('-','') for t in temp]
            else:
                temp.append(output)
    
    output_dict[entity_type].append(temp)
    return output_dict

def find_entity(data, entity, lang, samples):
    texts = []
    values = []
    counter = 0
    for datum in data:
        if lang!=datum['lang'] or entity not in datum['labels']: continue
        texts.append(datum['conversations'][0]['value'].replace('Passage: ','').split('\n\n')[0])
        values.append(datum['label_values'][entity])
        counter += 1
        if counter==samples: break
          
    return texts, values

def gpt_cleaner(queries, entity, samples=-1):
    queries = queries[0][:samples]
    t_dict = {entity: []}
#     print("queries", queries)
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)
        
    return t_dict

def gpt_cleaner(queries, entity, samples=-1):
    queries = queries[:samples]
    t_dict = {entity: []}
#     print("queries", queries)
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)
        
    return t_dict

def llama3_cleaner(queries, entity, samples=-1):
    queries = queries[entity]
    temp_dict = {entity: []}
    if entity=='Name':
        for query in queries:
            temp = []
            for qq in query:
                qq = qq.replace("'","")
                temp.append(qq)

            temp_dict[entity].append(temp)
    else:
        for query in queries:
            temp_dict[entity].append(query)
        
    return temp_dict

def gemini_cleaner(queries, entity='Age', samples=-1):
    query = queries
    print("QUERIES", queries)
    t_dict = {entity: []}
    query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
    query = query.replace('```','')
    query = query.replace('\n', '')
    query = query.replace('python','')
    query = query.replace('json','')
    query = query.replace('[[', '[')
    query = query.replace(']]', ']')
    query = query.replace('}]}', '}')
    if entity=='Money':
        query = query.replace('b', '')
        query = query.replace('m', '')
        query = query.replace('k', '')

    try:
        query = ast.literal_eval(query)[entity]
    except:
        query = ast.literal_eval(query)
#     print(query)
    if entity=='Money':
        t_dict[entity] = query
        temp = []
        for query in t_dict[entity]:
            temp.append(query.replace(',', ''))
            t_dict[entity] = temp
        return t_dict
    
    t_dict = postprocess_output_gpt(query, t_dict, entity)
    t_dict[entity] = t_dict[entity][0]
        
    return t_dict

def gemini_cleaner_list(queries, entity, samples=-1):
    queries = queries[0]
    t_dict = {entity: []}
    for query in queries:
        query = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", r"\\'", query)
        query = query.replace('```','')
        query = query.replace('json','')
        if 'It seems' in query or 'cannot' in query: 
            t_dict[entity].append([])
            continue
        query = ast.literal_eval(query)[entity]
        t_dict = postprocess_output_gpt(query, t_dict, entity)
        
    return t_dict

def ner_cleaner(queries, entity, lang, samples=-1):
    # Return a list of lists!
    queries = queries[lang][entity][:samples]
    t_dict = {entity: []}
    for query in queries: 
        # query is a list.
        t_dict = postprocess_output_ner(query, t_dict, entity)
    
    return t_dict[entity]
        
def ner_match_score(queries, answers, entity, samples=-1):
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0
    for query, answer in zip(queries[:samples], answers[:samples]):
#         print(query, answer)
        a = set(query)
        b = set(answer)
        c = a.intersection(b)
        tp += len(c)
        fn += len(b) - len(c)
        fp += len(a) - len(c)
        
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp


def ner_match_score_all(queries, answers, entity, samples=-1):
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0
    
    all_queries = []
    all_answers = []
    
    for query in queries[:samples]:
        all_queries.extend(query)
        
    for answer in answers[:samples]:
        all_answers.extend(answer)
    
    a = set(all_queries)
    b = set(all_answers)
    c = a.intersection(b)
    tp += len(c)
    fn += len(b) - len(c)
    fp += len(a) - len(c)
        
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp


def ner_match_score_gemini(queries, answers, entity, samples=-1):
#     print("\nIN IT")
#     print(queries)
    queries = queries[entity]
    tp = 0
    fn = 0
    fp = 0
    
    all_queries = []
    all_answers = []
    
    for query in queries:
        indices = [i for i, x in enumerate(queries) if x == query]
        indices = list(range(len(indices)))
        for index in indices:
            all_queries.append(f"{query}_{index}")
    
    
    temp_answers = []
    for answer in answers[:samples]:
        temp_answers.extend(answer)
        
    for answer in temp_answers:
        indices = [i for i, x in enumerate(temp_answers) if x == answer]
        indices = list(range(len(indices)))
        for index in indices:
            all_answers.append(f"{answer}_{index}")
    
    a = set(all_queries)
    b = set(all_answers)
    c = a.intersection(b)
    tp += len(c)
    fn += len(b) - len(c)
    fp += len(a) - len(c)
        
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall, tp, fn, fp

### NER extractor
def extract_entities_gemma(
    all_text,
    model_path,
    batch_size,
    entity,
    device,
):
    
    output_dict = {}
    output_dict[entity] = []
    
    def format_prompt(input):
        
        chat = [
                { "role": "user", 
                 "content": f"Please identify {entity} from the given text. Format the output as list with no additional text. Example: ['{entity} 1', '{entity} 2'].\n\nText: {input}" if entity!='Money' else f"Please identify Currency Value from the given text. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2'].\n\nText: {input}"},
                # { "role": "assistant", "content": "{output}" },
            ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(text)
        return text
    
    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False
                                  )
        # print(tokenized_text)
        return tokenized_text
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                  use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 device_map=device, torch_dtype=torch.bfloat16,)
                                                 # attn_implementation='flash_attention_2')
    model.eval()
    max_new_tokens=1024
    
    prompts = [format_prompt(text) for text in all_text]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)
    
    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        with torch.no_grad():
            prompt_batch.to(device)
            outputs = model.generate(**prompt_batch, 
                                     do_sample=False,
                                     max_length=max_new_tokens)
            outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            outputs = [output.replace("['", "") for output in outputs]
            outputs = [output.replace("']", "") for output in outputs]
            # print(outputs)
            outputs = [r.split('\nmodel\n')[-1].strip() for r in outputs]
            # print(outputs)
            output_dict = postprocess_output(outputs, output_dict, entity)

    
    # print(output_dict)
    return output_dict



def extract_entities_llama(
    all_text,
    model_path,
    batch_size,
    entity,
    device,
):
    
    output_dict = {}
    output_dict[entity] = []
    
    def format_prompt(input):
        conv = get_conv_template('llama-3')
        conv.set_system_message(
            f"Please identify {entity} from the given text. Format the output as list with no additional text. Example: ['{entity}_1', '{entity}_2']." if entity!='Money' else f"Please identify Currency Value from the given text. Format the output as list with no additional text. Example: ['Currency Value 1', 'Currency Value 2']."
        )
        conv.append_message(conv.roles[0], input)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()
        return text
    
    def collate_fn(batch):
        tokenized_text = tokenizer(batch,  
                                   return_tensors='pt', 
                                   add_special_tokens=False)
        return tokenized_text
        
        
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                  use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                 device_map=device, torch_dtype=torch.bfloat16,
                                                 attn_implementation='flash_attention_2')
    model.eval()
    max_new_tokens=1024
    
    prompts = [format_prompt(text) for text in all_text]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)
    
    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                         do_sample=False,
                                         max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [output.replace("['", "") for output in outputs]
                outputs = [output.replace("']", "") for output in outputs]
                # print(outputs)
                outputs = [r.split('assistant\n\n')[-1].strip() for r in outputs]
                # print(outputs)
                output_dict = postprocess_output(outputs, output_dict, entity)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")
    
    # print(output_dict)
    return output_dict

    
def extract_entities_LLM(
    all_text,
    model_path,
    batch_size,
    device,
    entity_type,
):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
#     print(all_text[:2])
    # Prepare input
    max_new_tokens = 1024
    output_dict = {}
    output_dict[entity_type] = []
    examples = [
        {
            "conversations": [
                {"from": "human", "value": f"Text: {text}"}, 
                {"from": "gpt", "value": "I've read this text."}, 
                {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                {"from": "gpt", "value": "[]"}
            ]
        } for text in all_text
    ]
    
    # Batch process helper
    def collate_fn(batch):
            tokenized_text = tokenizer(batch, 
#                                        padding=True, 
#                                        pad_to_multiple_of=8, 
                                       return_tensors='pt')
            return tokenized_text
        
    # Preprocess prompts...
    prompts = [preprocess_instance(example['conversations']) for example in examples]
    prompts = any2en(prompts)
    prompt_dataloader = DataLoader(prompts, collate_fn=collate_fn, batch_size=batch_size)
    
    for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
        try:
            with torch.no_grad():
                prompt_batch.to(device)
                outputs = model.generate(**prompt_batch, 
                                         do_sample=False,
                                         max_length=max_new_tokens)
                outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                outputs = [r.split('ASSISTANT:')[-1].strip() for r in outputs]
                output_dict = postprocess_output(outputs, output_dict, entity_type)
        except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")
        
    return output_dict

def extract_entities_LLM_gen(all_text,
                               model,
                               tokenizer,
                               batch_size,
                               max_new_tokens,
                               device,
                               list_of_entities=['Name','Age','Money','Zipcode','SSN','Date'],
                            ):
    output_dict = {}
    raw_entity_mapping = []
    for entity_type in list_of_entities:
        # print(all_text)
        output_dict[entity_type] = []
        examples = [{"conversations": [
                {"from": "human", "value": f"Text: {text}"}, 
                {"from": "gpt", "value": "I've read this text."}, 
                {"from": "human", "value": f"What describes {entity_type} in the text?"}, 
                {"from": "gpt", "value": "[]"}
            ]} for text in all_text]
        
        def collate_fn(batch):
            tokenized_text = tokenizer(batch, 
#                                        padding=True, 
#                                        pad_to_multiple_of=8, 
                                       return_tensors='pt')
            return tokenized_text
            
        # Preprocess prompts...
        prompts = [preprocess_instance(example['conversations']) for example in examples]
        prompts = any2en(prompts)
        prompt_dataloader = DataLoader(prompts, collate_fn = collate_fn, batch_size=batch_size)
        # print("\n", entity_type)
        # UniNER
        for i, prompt_batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            try:
                with torch.no_grad():
                    # print(i, prompt_batch)
                    # prompt_batch = tokenizer(prompt_batch, return_tensors='pt')
                    prompt_batch.to(device)
                    outputs = model.generate(**prompt_batch, do_sample=False,
                                             max_length=max_new_tokens, 
                                            )
                    outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    #                 print(outputs, '\n')
                    outputs = [r.split('ASSISTANT:')[-1].strip() for r in outputs]
                    if entity_type=="Name" or entity_type=="Full Name":
                        outputs = clean_suffix_name(outputs) 
                    if entity_type=="Age":
                        outputs = [str(re.findall(r"\b(\d{1,3})\b", output)) for output in outputs]
                    if entity_type=="Zipcode":
                        outputs = [str(re.findall(r"\b(\d{5}(?:-\d{4})?)\b", output)) for output in outputs]
                    if entity_type=="Money":
                        raw_entity_mapping.extend([re.findall(r"\d+(?:,\d+)?", output) for output in outputs])
                        outputs = [output.replace(",", "") for output in outputs]
                        outputs = [str(re.findall(r"[-+]?(?:\d*\.*\d+)", output)) for output in outputs]
                    for output in outputs:
                        if len(output) > 0: 
                            if entity_type=="Name" or entity_type=="Full Name":
                                temp = unicodedata.normalize("NFKD", ast.literal_eval(output))
                                temp = temp.replace('[', '')
                                temp = temp.replace(']', '')
                                temp = temp.split(", ")
                                output_dict[entity_type].append(temp)
                            elif entity_type=="Money":
                                temp = ast.literal_eval(output)
                                temp = [t.replace('-','') for t in temp]
                                output_dict[entity_type].append(temp)
                            else:
                                output_dict[entity_type].append(ast.literal_eval(output))
            except:
                traceback.print_exc()
                print(f"Error on batch ID: {i}")
                
            
        gc.collect()
        torch.cuda.empty_cache()
    # print("\n OUTPUT:", output_dict)
    # return raw_entity_mapping, output_dict
    list_of_output_dicts = [{} for i in range(len(output_dict[list_of_entities[0]]))]
    for i in range(len(output_dict[list_of_entities[0]])):
        for entity in list_of_entities:
            list_of_output_dicts[i][entity] = output_dict[entity][i] # unicodedata.normalize("NFKD", output_dict[entity][i])

    if len(list_of_entities)==1 and list_of_entities[0]!='Money':
        raw_entity_mapping = [[] for i in range(len(list_of_output_dicts))]

    # print(list_of_output_dicts)
    
    return raw_entity_mapping, list_of_output_dicts

# OPUS-MT translation
def translate_OPUS_MT(all_text, model_path, device='cuda:0'):
    print(model_path)
    translated_de_encrypted_en = []
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    temp_model = MarianMTModel.from_pretrained(model_path)
    temp_model.to(device)
    temp_model.eval()
    text = any2en(all_text)
    def collate_fn(batch):
        tokenized_text = tokenizer(batch, 
                                   padding=True, 
                                   return_tensors='pt')
        return tokenized_text
        
    prompt_dataloader = DataLoader(text, collate_fn = collate_fn, batch_size=1)
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(prompt_dataloader), total=len(prompt_dataloader)):
            try:
                batch.to(device)
                outputs = temp_model.generate(**batch, 
                                              do_sample=False,
                                             )
                translated_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                translated_de_encrypted_en.extend(translated_text)
            except Exception as e:
                translated_de_encrypted_en.append("Error")
    
    return translated_de_encrypted_en
            
# GPT-4 Translation
def translate_GPT(all_text, target_lang, api_key):
    openai.api_key = api_key
#     entities = ", ".join(list_of_entities)
    async def query_llm(text, model="gpt-4o"):
        response = openai.ChatCompletion.create(
          model=model,
          messages=[
            {
              "role": "system",
                # Prompt format from Ye et al. https://arxiv.org/pdf/2303.10420.pdf
              "content": f"You will be provided with sentences in English, and your task is to translate them into {target_lang}."
            },
            {
              "role": "user",
              "content": text 
            },
          ],
          temperature=0,
          max_tokens=1024,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        return response
    
    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response['choices'][0]['message']['content']
            response = response.encode().decode()
            return response
        except Exception as e:
#             print(e)
#             return response
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en

# Gemini NER
def translate_gemini(all_text, target_lang, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=4096,
        top_p=1,
    )
    
    sleep_time = 600
    def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"You will be provided with sentences in English, and your task is to translate them into {target_lang}. Return them as a list of strings and nothing else."
            ]
        )
        
        temp_text = text
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
#         text = response
        return text
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    # afk = gemini_cleaner(query_llm(arr))
    translated_de_encrypted_en = query_llm(arr)
    translated_de_encrypted_en = translated_de_encrypted_en.replace("```","")
    translated_de_encrypted_en = translated_de_encrypted_en.replace("python","")
    translated_de_encrypted_en = ast.literal_eval(translated_de_encrypted_en)
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en

# GPT-4 NER
def extract_entities_GPT(all_text, entity, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    openai.api_key = api_key
#     entities = ", ".join(list_of_entities)
    async def query_llm(text, model="gpt-4-turbo"):
        response = openai.ChatCompletion.create(
          model=model,
          messages=[
            {
              "role": "system",
                # Prompt format from Ye et al. https://arxiv.org/pdf/2303.10420.pdf
              "content": f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}"
            },
            {
              "role": "user",
              "content": text 
            },
          ],
          temperature=0,
          max_tokens=1024,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0
        )
        return response
    
    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response['choices'][0]['message']['content']
            response = response.encode().decode()
            return response
        except Exception as e:
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en, entity

# Claude NER
def claude_cleaner(content, lang, entity):
    if 'Sorry' in content or 'no' in content or 'apologize' in content:
        return []
    values = content.split(":")
    if len(values)==1: return []
    values = values[1]
    values = values.split("\n\n")[1]
    values = values.split('\n')
    temp = []
    for value in values:
        if entity=='Name':
            pattern = r'[a-zA-Z]+'
            value = re.findall(pattern, value)
            value = " ".join(value)
        if entity=='Money' or entity=='Age':
            pattern = r'[a-zA-Z]*(\d{1,10}(?:,\d{3})*(?:\.\d+)?)[a-zA-Z]*'
            value = re.findall(pattern, value)
            if len(value)==0:
                value = ""
            else:
                value = value[-1]
                value = value.replace(',','')
#         print("HERE", value)

        temp.append(value)

    return temp
    
def load_claude_data(lang, entity):
    temp = []
    for i in range(1, 301):
        path = f'new_data/{lang}_{entity}/{i}.txt' if lang!='english' or entity!='Name' else f'new_data/{lang}_{entity}/{lang}_{entity}_test_{i}.txt'
        if os.path.exists(path):
            data = load_json_dataset(path)
            temp.append(claude_cleaner(data['content'][0]['text'], lang, entity))
        else:
            temp.append([])
        
    return {entity: temp}
    
def extract_entities_claude(all_text, entity, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import anthropic
    
    async def query_llm(text, model="claude-3-opus-20240229"):
        client = anthropic.Anthropic(
            # defaults to os.environ.get("ANTHROPIC_API_KEY")
            api_key=api_key,
        )
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            system = f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text
                        }
                    ]
                }
            ]
        )
        response = message.content
        return response
    
    async def extract_text(text, idx, crash_array):
        try:
            response = await query_llm(text)
            response = response['choices'][0]['message']['content']
            response = response.encode().decode()
            return response
        except Exception as e:
#             print(e)
#             return response
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],
                                                      return_exceptions=True))
    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en, entity

# Gemini NER
def extract_entities_gemini(all_text, entity, lang, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=4096,
        top_p=1,
    )
    
    sleep_time = 600
    def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}." if entity!='Money' else f"Please identify {entity} from the given text. Do not provide the currency, only provide the value. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}."
            ]
        )
        
        temp_text = text
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
#         text = response
        return text
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = query_llm(arr)

    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en, entity

def extract_entities_gemini_async(all_text, entity, lang, api_key, list_of_entities=['Name','Age','Money','Zipcode','SSN','Date']):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    generation_config = genai.types.GenerationConfig(
        temperature=0,
        max_output_tokens=1024,
        top_p=1,
    )
    
    sleep_time = 600
    async def query_llm(text, model="gemini-1.5-pro-001"):
        model = genai.GenerativeModel(
            model_name = model,
            generation_config=generation_config,
            system_instruction=[
                f"Please identify {entity} from the given text. Format the output as a dictionary of lists: {{'{entity}': ['{entity}_1', '{entity}_2']}}."
            ]
        )
        
        temp_text = [f"{text}"]
        response = model.generate_content(temp_text)
        text = response._result.candidates[0].content.parts[0].text
        return text
    
    async def extract_text(text, idx, crash_array, sleep_time=600):
        try:
            response = await query_llm(text)
#             response = response['choices'][0]['message']['content']
#             response = response.encode().decode()
            return response
        except Exception as e:
            print(e)
            return response
            print(f"Timed out: Index {idx}. Retrying...")
            crash_array.append(idx)
            response = await extract_text(text, idx, crash_array)
            return response
        
    crash_arrays = []
    loop = asyncio.get_event_loop()
    arr = all_text
    counter = range(len(arr))
    crash_array = []
    a = time.time()
    print(f"Starting API NER extraction. Time: {a}")
    translated_de_encrypted_en = loop.run_until_complete(asyncio.gather(*[extract_text(arr[i], i, crash_array) for i in counter],return_exceptions=True))

    print(f"Task target:{len(arr)} Time taken: {time.time()-a}")
    crash_arrays.append(crash_array)
    print(f"Crash arrays:", crash_arrays)
    
    return translated_de_encrypted_en, entity

### Sanitation
def M_epsilon(x, n_lower, n_upper, epsilon, discretization_size=100):
  # Sample within given range to provide privacy guarantee
  n_upper = int(n_upper)
  n_lower = int(n_lower)
  total_range = n_upper-n_lower
  x = (x-n_lower)*discretization_size/total_range
  p_i = []
  for s in range(discretization_size):
      p_i.append(math.exp(-abs(x-s)*epsilon/2))
  p_i = [val/sum(p_i) for val in p_i]
  noised_output = np.random.choice(range(discretization_size),1,p=p_i)*total_range/discretization_size+n_lower
  return int(noised_output[0])


def get_encrypted_values(inputs, entity, N, rho, epsilon, c):
    # print("Pre-encryption:", entities)
    # inputs is a list of lists.
    new_entities = []
    if entity=='Name':
        for input in inputs:
            temp = []
            for k in range(len(input)):
                temp_name = names.get_full_name(gender='male')
                while temp_name in temp:
                    temp_name = names.get_full_name(gender='male')
                temp.append(temp_name)
            # entities['Full Name'] = rng.generate(descent=rng.Descent.ENGLISH, sex=rng.Sex.MALE, limit=1)[0]
            new_entities.append(temp)

    # Add noise to age and encrypt
    if entity=='Age':
        for input in inputs:
            temp = []
            for real_age in input:
                temp_age = M_epsilon(int(real_age),10,99,epsilon)
                # while temp_age==int(real_age):
                #     temp_age = M_epsilon(int(real_age),10,99,epsilon)
                age = str(temp_age)
                temp.append(age)

            new_entities.append(temp)
            
    # Account for currency symbols. UniNER returns '$' along with the amount.
    if entity=='Money':
        counter = 0
        for input in inputs:
            temp = []
            for real_money in input:
                # money = real_money
                try:
                    # while money==real_money:
                    #     if int(float(real_money))==1:
                    #         money = round(M_epsilon(int(float(real_money)),1,2,epsilon), 2)
                    #     elif int(float(real_money))<100:
                    #         money = round(M_epsilon(int(float(real_money)),2,1000,epsilon), 2)
                    #     elif int(float(real_money))<10000:
                    #         money = round(M_epsilon(int(float(real_money)),100,100000,epsilon), 2)
                    #     else:
                    #         money = round(M_epsilon(int(float(real_money)),10000,100000,epsilon), 2)
                    money = M_epsilon(int(float(real_money)),
                                      max(0,int(float(real_money)*(1-rho['Money']))),
                                      min(int(float(real_money)*(1+rho['Money'])),int(float(real_money)*2)),
                                      epsilon,
                                     )   
                    temp.append(str(money))
                except Exception as e:
                    counter += 1
                    print("COUNTER:", counter)
                    print(e)
                    print("Error value:", real_money)
                    temp.append('NaN')
            
            new_entities.append(temp)
        
    return new_entities