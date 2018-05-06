#!/usr/bin/env bash

# DATASET='catsDogs'
# DATASET='ATVS'
DATASET='Warsaw'
# DATASET='MobBioFake'

################################
# ============ Settings:
PERIMGSTD=False # default if to compare with slim inception preprocessing
CHANNELS=3 # default if to compare with slim inception preprocessing
ENCODE=JPEG

TRAINSTEPS=2000
LR=0.001
WD=0.0004
SZ=112
VGGRESIZE=$SZ

################################
#PREPROCESS=y_vgg
PREPROCESS=y_combined # like y_vgg + distort brightness and contrast

VGGRESIZE=128
VGGUSEASPECTPRES=False
VGGSUBMEAN=False

##################################
MODEL1=spoofnet_y_BN_noLRN

################## /usr/bin/
# python ./pipeline_tf/train_image_classifier_y.py \
#             --dataset_name=$DATASET \
#             --model_name=$MODEL1 \
#             --preprocessing_name=$PREPROCESS \
#             --image_size=$SZ \
#             --encode_type=$ENCODE \
#             --vgg_resize_side=$VGGRESIZE \
#             --vgg_use_aspect_preserving_resize=$VGGUSEASPECTPRES \
#             --vgg_sub_mean_pixel=$VGGSUBMEAN \
#             --initial_learning_rate=$LR \
#             --weight_decay=$WD \
#             --per_image_standardization=$PERIMGSTD \
#             --channels=$CHANNELS \
#             --max_number_of_steps=$TRAINSTEPS \
#             --save_interval_secs=60 \
#             --save_summaries_secs=60 \
#             --log_every_n_steps=10 \
#             --validation_every_n_steps=100  \
#             --test_every_n_steps=200

python ./pipeline_tf/eval_image_classifier_y.py \
            --dataset_name=$DATASET \
            --dataset_split_name_y='train' \
            --dataset_split_name_y2='validation' \
            --use_placeholders=False \
            --eval_batch_size=100 \
            --model_name=$MODEL1 \
            --preprocessing_name=$PREPROCESS \
            --image_size=$SZ \
            --encode_type=$ENCODE \
            --vgg_resize_side=$VGGRESIZE \
            --vgg_use_aspect_preserving_resize=$VGGUSEASPECTPRES \
            --vgg_sub_mean_pixel=$VGGSUBMEAN \
            --initial_learning_rate=$LR \
            --weight_decay=$WD \
            --per_image_standardization=$PERIMGSTD \
            --channels=$CHANNELS

# python ./pipeline_tf/eval_image_classifier_y.py \
#             --dataset_name=$DATASET \
#             --dataset_split_name_y='train' \
#             --dataset_split_name_y2='validation' \
#             --use_placeholders=True \
#             --oversample_at_eval=True \
#             --eval_batch_size=10 \
#             --model_name=$MODEL1 \
#             --preprocessing_name=$PREPROCESS \
#             --image_size=$SZ \
#             --encode_type=$ENCODE \
#             --vgg_resize_side=$VGGRESIZE \
#             --vgg_use_aspect_preserving_resize=$VGGUSEASPECTPRES \
#             --vgg_sub_mean_pixel=$VGGSUBMEAN \
#             --initial_learning_rate=$LR \
#             --weight_decay=$WD \
#             --per_image_standardization=$PERIMGSTD \
#             --channels=$CHANNELS

