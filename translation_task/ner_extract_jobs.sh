mkdir results
# MODEL=/PATH/TO/uniner-7b-pii-v3
# MODEL=/PATH/TO/Meta-Llama-3-8B-Instruct
# MODEL=gpt-4
# MODEL=gemini
# MODEL=claude
MODEL=/PATH/TO/gemma-2-9b-it
DATA=datasets/pii-masking-200k/pii_masking_200k_en_fr_de_test_v3.json
API_KEY=SECRET_KEY
SAMPLES=300
ENTITY=Name
LANG=all
python ner_comparison.py \
    --device 0 \
    --data_path ${DATA} \
    --ner_path ${MODEL} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    --entity ${ENTITY} \
    --lang ${LANG}