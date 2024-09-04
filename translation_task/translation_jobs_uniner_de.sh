DEVICE=0
NER_MODEL=/PATH/TO/models/uniner-7b-pii-v3
# NER_MODEL=/PATH/TO/models/gemma-2-9b-it
# NER_MODEL=/PATH/TO/Meta-Llama-3-8B-Instruct
TASK=de-en
ENTITY=Age
TRANS_MODEL=opus-mt-de-en
# TRANS_MODEL=gpt-4
# TRANS_MODEL=gemini
SAMPLES=1
# API_KEY=SECRET_KEY
API_KEY=SECRET_KEY

TAG=./SAVE_PATH_${TASK}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 translation_task.py \
    --task ${TASK} \
    --entity ${ENTITY} \
    --translation_model ${TRANS_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    

TASK=de-en
ENTITY=Name
# TRANS_MODEL=opus-mt-fr-en
TRANS_MODEL=opus-mt-de-en
# TRANS_MODEL=gpt-4
# TRANS_MODEL=gemini
SAMPLES=1
# API_KEY=SECRET_KEY
API_KEY=SECRET_KEY

TAG=./SAVE_PATH_${TASK}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 translation_task.py \
    --task ${TASK} \
    --entity ${ENTITY} \
    --translation_model ${TRANS_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \
    
TASK=de-en
ENTITY=Money
# TRANS_MODEL=opus-mt-fr-en
TRANS_MODEL=opus-mt-de-en
# TRANS_MODEL=gpt-4
# TRANS_MODEL=gemini
SAMPLES=1
# API_KEY=SECRET_KEY
API_KEY=SECRET_KEY

TAG=./SAVE_PATH_${TASK}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 translation_task.py \
    --task ${TASK} \
    --entity ${ENTITY} \
    --translation_model ${TRANS_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --tag ${TAG} \
    --samples ${SAMPLES} \
    --api_key ${API_KEY} \