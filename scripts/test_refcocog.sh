#!/bin/bash
uname -a
#date
#env
date

DATASET=refcocog
DATA_PATH=datasets
REFER_PATH=refcocog
BERT_PATH=pretrained_weights/bert-base-uncased/
MODEL=srun
SWIN_TYPE=base
IMG_SIZE=448
ROOT_PATH=SRUN_output_refcocog
RESUME_PATH=${ROOT_PATH}/model_best_${DATASET}.pth
OUTPUT_PATH=${ROOT_PATH}/${DATASET}
SPLIT=test

CUDA_VISIBLE_DEVICES=0 python eval.py --model ${MODEL} --swin_type ${SWIN_TYPE} \
        --dataset ${DATASET} --split ${SPLIT} \
        --img_size ${IMG_SIZE} --resume ${RESUME_PATH} \
        --bert_tokenizer ${BERT_PATH} --ck_bert ${BERT_PATH} \
        --refer_data_root ${DATA_PATH} --refer_root ${REFER_PATH}
