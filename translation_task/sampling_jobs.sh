DEVICE=0
NER_MODEL=/PATH/TO/NER/MODEL # uniner-7b-pii
TASK=fr-en
ENTITY=Money
TRANS_MODEL=opus-mt-fr-en
# TRANS_MODEL=gpt-4
SAMPLES=50
API_KEY=SECRET_KEY

TAG=/SAVE/PATH/${TASK}_${ENTITY}_NER_uniner_TRANS_${TRANS_MODEL}_samples_${SAMPLES}

CUDA_VISIBLE_DEVICES=${DEVICE} python3 collect_samples_v2.py \
    --task ${TASK} \
    --entity ${ENTITY} \
    --translation_model ${TRANS_MODEL} \
    --batch_size 1 \
    --ner_path ${NER_MODEL} \
    --tag ${TAG} \
    --api_key ${API_KEY}