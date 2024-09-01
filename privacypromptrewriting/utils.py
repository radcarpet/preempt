import sys
sys.path.insert(0, '/path/to/universal-ner/src')

import random_name_generator as rng
import math
import numpy as np
from pyfpe_ff3 import FF3Cipher
import re
import datetime

key = "EF4359D8D580AA4F7F036D6F04FC6A94"
tweak = "D8E7920AFA330A73"

def format_align_digits(text, reference_text):
    if len(text) != len(reference_text):
        for idx, t in enumerate(reference_text):
            if not t.isdigit():
                text = text[:idx] + reference_text[idx] + text[idx:]
    return text


def extract_entities_regex(text):
  output_dict = {}
  # Extract name (assuming it's in "First Last" format)
  name_pattern = r"([A-Z][a-z]+ [A-Z][a-z]+)"
  name_match = re.search(name_pattern, text)
  if name_match:
      output_dict['name'] = name_match.group(1)
 # else:
 #     name = None

  # Extract age
  age_pattern = r"\b(\d{1,3})\s*years? old\b"
  age_match = re.search(age_pattern, text)
  if age_match:
      output_dict['age'] = age_match.group(1)
  #else:
  #    age = None

  # Extract salary
  salary_pattern = r"\$([0-9,]+)"
  salary_match = re.search(salary_pattern, text)
  if salary_match:
      output_dict['salary'] = salary_match.group(1).replace(',', '')
  #else:
  #    salary = None

  # Extract zipcode
  #zipcode_pattern = r"\b(\d{5}(?:-\d{4})?)\b"
  #zipcode_match = re.search(zipcode_pattern, text)
  #if zipcode_match:
  #    output_dict['zipcode'] = zipcode_match.group(1)
  #else:
  #    zipcode = None

  # Extract SSN
  ssn_pattern = r"\b(\d{3}-\d{2}-\d{4})\b"
  ssn_match = re.search(ssn_pattern, text)
  if ssn_match:
      output_dict['ssn'] = ssn_match.group(1)
  #else:
  #    ssn = None

  # Extract date in yyyy-mm-dd format
  date_pattern = r"(\d{4}-\d{2}-\d{2})"
  date_match = re.search(date_pattern, text)
  if date_match:
      output_dict['date'] = date_match.group(1)
  #else:
  #    date = None
  return output_dict


def extract_entities_LLM(text,model,list_of_entities=['Full Name','Age','Money','Zipcode','SSN','Date']):
  output_dict = {}
  for entity_type in list_of_entities:
    example = {"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}
    prompt = preprocess_instance(example['conversations'])
    output_dict[entity_type] = (model(prompt, max_length=max_new_tokens, return_full_text=False))
  return output_dict

def convert_date_to_numeric(date_str):
    # Define a mapping of month names to numerical values
    month_mapping =month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    words = date_str.split()
    month = month_mapping.get(words[0])
    # Remove "st", "nd", "rd", or "th" from the day
    day = int(words[1][:-2])  
    date_object = datetime.datetime(datetime.datetime.now().year, month, day)
    day_of_year = date_object.timetuple().tm_yday
    return day_of_year

def convert_day_of_year_to_date(day_of_year, year=2023):
    january_1st = datetime.datetime(year, 1, 1)
    target_date = january_1st + datetime.timedelta(days=day_of_year - 1)
    formatted_date = target_date.strftime("%B %d")
    return formatted_date

def split_email(email):
    at_index = email.find('@')
    username = email[:at_index]
    domain = email[at_index:]
    return username, domain

def format_new_number(phone_number, new_number):
    # Extract all non-numeric characters
    non_numeric_chars = ''.join(char for char in phone_number if not char.isdigit())
    reintroduced_number = ''
    non_numeric_index = 0
    numeric_index = 0

    for char in phone_number:
        if not char.isdigit():
            reintroduced_number += non_numeric_chars[non_numeric_index]
            non_numeric_index += 1
        else:
            reintroduced_number += new_number[numeric_index]
            numeric_index += 1

    return reintroduced_number

def separate_and_reintroduce(original_string, encrypted_string):
    non_alphanumeric_chars = ''.join(char for char in original_string if not char.isalnum())
    reintroduced_string = ''
    non_alphanumeric_index = 0
    alphanumeric_index = 0
    
    for char in original_string:
        if not char.isalnum():
            reintroduced_string += non_alphanumeric_chars[non_alphanumeric_index]
            non_alphanumeric_index += 1
        else:
            reintroduced_string += encrypted_string[alphanumeric_index]
            alphanumeric_index += 1

    return reintroduced_string

  
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


def generate_encrypted_entities_regex(entities, N, epsilon, c):
  
  output_dict = {}
  if 'name' in entities.keys():
    output_dict['name'] = rng.generate(descent=rng.Descent.ENGLISH, sex=rng.Sex.MALE, limit=1)[0]

  # Add noise to age and encrypt
  if 'age' in entities.keys():
    output_dict['age'] = M_epsilon(int(entities['age']),10,99,epsilon)
  #age = str(np.random.choice(100,1,age_probs)[0])
  #age = c.encrypt(age)

  #Sample Salary on restricted domain (Tunable FPE not yet available)
  if 'salary' in entities.keys():
    output_dict['salary'] = M_epsilon(
      int(entities['salary']),
      N['salary']/100,
      N['salary'],
      epsilon
    )

  # FPE on Zipcode
  #if 'zipcode' in entities.keys():
  #  output_dict['zipcode'] = c.encrypt(entities['zipcode'])

  # FPE on SSN while preserving dash structure
  if 'ssn' in entities.keys():
    output_dict['ssn'] = format_align_digits(c.encrypt(entities['ssn'].replace("-", "")),entities['ssn'])
  return output_dict#{"name": name, "age": age, "salary": salary, "zipcode": zipcode, "ssn": ssn}

def generate_encrypted_entities_LLM(entities, N, rho, epsilon, c, key, tweak):
  
  if 'Full Name' in entities.keys():
    entities['Full Name'] = rng.generate(descent=rng.Descent.ENGLISH, sex=rng.Sex.MALE, limit=1)[0]

  # Add noise to age and encrypt
  if 'age' in entities.keys():
    age = M_epsilon(int(entities['age']),10,99,epsilon)
    entities['Age'] = age #c.encrypt(age) we got rid of it cause FPE breaks on small numbers

  # Sample money on restricted domain (Tunable FPE not yet available)
  if 'Money' in entities.keys(): 
    pattern = re.compile(r'\$?(\d+)')
    matches = pattern.findall(entities['Money'])
    money = int(matches[0])
    entities['Money'] = M_epsilon(
      money,
      N/100,
      N,
      epsilon
    )
  # FPE on Zipcode
  if 'Zipcode' in entities.keys():
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=10)
    entities['Zipcode'] = c.encrypt(entities['Zipcode'])

  # FPE on SSN while preserving dash structure
  if 'SSN' in entities.keys():  
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=10)
    entities['SSN'] = format_align_digits(c.encrypt(entities['SSN'].replace("-", "")),entities['SSN'])
  
  if 'Date' in entities.keys():
    date = entities['Date']
    numeric_date = convert_date_to_numeric(date)
    numeric_date = M_epsilon(numeric_date, -365, 730, epsilon) % 365
    entities['Date'] = convert_day_of_year_to_date(numeric_date)
    
  if 'Email' in entities.keys():
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=36)
    username, domain = split_email(entities['Email'])
    new_username = c.encrypt(username)
    entities['Email'] = new_username+domain
    
  if 'Phone Number' in entities.keys():
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=10)
    extracted_number = ''.join(char for char in entities['Phone Number'] if char.isdigit())
    new_number = c.encrypt(extracted_number)
    entities['Phone Number'] = format_new_number(entities['Phone Number'],new_number)
    
  if 'Product ID' in entities.keys():
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=36)
    product_id = ''.join(char for char in entities['Product ID'] if char.isalnum())
    encrypted_id = c.encrypt(product_id)
    entities['Product ID'] = separate_and_reintroduce(entities['Product ID'],encrypted_id)
    
  if 'Order ID' in entities.keys():
    c = FF3Cipher(key, tweak,allow_small_domain=True, radix=36)
    order_id = entities['Order ID']
    entities['Order ID'] = c.encrypt(order_id)
    
  return entities


#def decrypt_entities_regex(entities, original_entities, c):
#  # Recover original name
#  name = original_entities['name']
#
#  # Decrypt age
#  age = c.decrypt(entities['age'])
#
#  # Recover original Salary
#  salary = original_entities['salary']
#
#  # Decrypt Zipcode
#  zipcode = c.decrypt(entities['zipcode'])
#
#  # Decrypt SSN while preserving dash structure
#  ssn = format_align_digits(c.decrypt(entities['ssn'].replace("-", "")),entities['ssn'])
#  return {"name": name, "age": age, "salary": salary, "zipcode": zipcode, "ssn": ssn}
#
#def decrypt_entities_LLM(entities, original_entities, c):
#  # Recover original name
#  if 'Full Name' in entities.keys():
#    entities['Full Name'] = original_entities['Full Name']
#
#  # Decrypt age
#  if 'Age' in entities.keys():
#    entities['Age'] = original_entities['Age'] #c.decrypt(entities['Age']) #got rid of decrypt with lookupt table cause FPE breaks on small numbers
#
#  # Recover original Salary
#  if 'Money' in entities.keys():
#    entities['Money'] = original_entities['Money']
#
#  # Decrypt Zipcode
#  if 'Zipcode' in entities.keys():
#    entities['Zipcode'] = c.decrypt(entities['Zipcode'])
#
#  # Decrypt SSN while preserving dash structure
#  if 'SSN' in entities.keys():
#    entities['SSN'] = format_align_digits(c.decrypt(entities['SSN'].replace("-", "")),entities['SSN'])
#
#  # Decrypt Date
#  #if 'Date' in entities.keys():
#    # work in progress, many possible standards for formatting with unclear preservation during NER, unclear what type of encryption/noising scheme we would want to employ for this
#  return entities

def update_entities_regex(text, entities):
  # Replace extracted values in the original text
  if 'name' in entities.keys():
    text = re.sub(r"([A-Z][a-z]+ [A-Z][a-z]+)", entities['name'], text)
  if 'age' in entities.keys():
    text = re.sub(r"\b(\d{1,3})\s*years? old\b", str(entities['age']) + " years old", text)
  if 'salary' in entities.keys():
    text = re.sub(r"\$([0-9,]+)", "$"+f"{int(entities['salary']):,}", text)
  #if 'zipcode' in entities.keys():
  #  text = re.sub(r"\b(\d{5}(?:-\d{4})?)\b", str(entities['zipcode']), text)
  if 'ssn' in entities.keys():
    text = re.sub(r"\b(\d{3}-\d{2}-\d{4})\b", entities['ssn'], text)
  return text

def update_entities_LLM(text, encrypted_entities, decrypted_entities):
  for entity in encrypted_entities.keys():
    text = text.replace(encrypted_entities[entity], decrypted_entities[entity])
