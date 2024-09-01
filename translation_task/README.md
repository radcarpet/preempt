# Translation

- Generate samples for named entity with the following:

```
python3 collect_samples.py --device {DEVICE_ID} \
                           --segment {DATA_PARTITION} \
                           --batch_size 64 \
                           --split train \
                           --entity {NAMED_ENTITY} \
                           --samples 1000 \
                           --task {TASK} \
```
