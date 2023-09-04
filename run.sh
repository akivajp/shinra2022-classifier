#!/bin/bash

set -eu
#set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE:-${(%):-%N}}")"; pwd)"                                                       
source ${script_dir}/common.sh 

# 出力キー
DEFAULT_OUTPUT_KEY=$(date "+AUTO.AIP.%Y%m")
OUTPUT_KEY=${OUTPUT_KEY:-${DEFAULT_OUTPUT_KEY}} 

# 出力ディレクトリ
OUTPUT_DIR=${OUTPUT_DIR:-output}

# モデルパラメータ
EMBED_SIZE=${EMBED_SIZE:-200}
FEATURE_VOCAB_SIZE=${FEATURE_VOCAB_SIZE:-10000}
ENTITY_EMBED_SIZE=${ENTITY_EMBED_SIZE:-200}
HIDDEN_SIZE=${HIDDEN_SIZE:-200}

# 並列処理設定
NUM_WORKERS=${NUM_WORKERS:-2}

# クロスバリデーション設定
CV_FOLD=${CV_FOLD:-5}
CV_EPOCHS=${CV_EPOCHS:-10}

# 訓練設定
TRAIN_EPOCHS=${TRAIN_EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-100}

# GPU設定 (GPU番号、0オリジン、-1以下ならGPU不使用)
USE_GPU=${USE_GPU:-0}

# 推論設定
N_BEST=${N_BEST:-5}

if [ $# -lt 6 ]; then
    echo "Usage: $0" \
        "{path to base cirrussearch-content.json.gz}" \
        "{path to new cirrussearch-content.json.gz}" \
        "{path to base cirrussearch-general.json.gz}" \
        "{path to new cirrussearch-general.json.gz}" \
        "{path to annotation jsonl file}" \
        "{path to ENE definition jsonl file}"
    exit 1
fi

base_cirrus_content_path=$1
new_cirrus_content_path=$2
base_cirrus_general_path=$3
new_cirrus_general_path=$4
annotation_jsonl_path=$5
ene_definition_jsonl_path=$6
vec_dir=${OUTPUT_DIR}/vectors
base_corpus=${vec_dir}/base-corpus.txt
new_corpus=${vec_dir}/new-corpus.txt
merged_corpus=${vec_dir}/merged-corpus.txt
entity_embed_file=${vec_dir}/entity_vectors.txt
classification_dir=${OUTPUT_DIR}/classification
base_all_dataset=${classification_dir}/dataset-base-all.jsonl
base_labeled_dataset=${classification_dir}/dataset-base-labeled.jsonl
new_nolabel_dataset=${classification_dir}/dataset-new-nolabel.jsonl
merged_all_dataset=${classification_dir}/dataset-merged-all.jsonl
vocab_file=${classification_dir}/vocab.json
cross_validation_dir=${classification_dir}/cross-validation
training_dir=${classification_dir}/training
model_file=${training_dir}/epoch-$(printf "%03d" ${TRAIN_EPOCHS}).model

show-exec mkdir -p ${vec_dir}
show-exec mkdir -p ${classification_dir}
# 旧バージョンのCirrusSearch Content Dumpに "base" タグを付けてコーパスを作成
show-exec MECABRC=/etc/mecabrc python3 ./WikiEntVec/make_corpus.py \
    --cirrus_file ${base_cirrus_content_path} \
    --output_file ${base_corpus} \
    --tag base \
    --tokenizer mecab \
    --tokenizer_option \"-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd\"
# 新バージョンのCirrusSearch Content Dumpに "new" タグを付けてコーパスを作成
show-exec MECABRC=/etc/mecabrc python3 ./WikiEntVec/make_corpus.py \
    --cirrus_file ${new_cirrus_content_path} \
    --output_file ${new_corpus} \
    --tag new \
    --tokenizer mecab \
    --tokenizer_option \"-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd\"
# 旧・新バージョンのコーパスをマージ
show-exec-pv paste -d '"\\n"' ${base_corpus} ${new_corpus} \
    \| pv -cl \
    \| awk '"\$0"' \
    \| pv -cl \
    \> ${merged_corpus}
 エンティティベクトルの学習
show-exec python3 ./WikiEntVec/train.py \
    --corpus_file ${merged_corpus} \
    --output_dir ${vec_dir} \
    --embed_size ${EMBED_SIZE} \
    --workers ${NUM_WORKERS}
# アノテーションファイルからアノテーションキーを取得
annotation_keys=$(
    cat ${annotation_jsonl_path} \
        | head -n 100 \
        | jq -sr '[.[].ENEs | keys] | flatten | unique | .[]'
)
show-info "annotation_keys: ${annotation_keys}"
# 旧バージョンのWikipedia記事とアノテーションファイルを元にデータセットを作成
show-exec MECABRC=/etc/mecabrc python3 ./shinra-classify/make_dataset.py \
    --cirrus_content_file ${base_cirrus_content_path} \
    --cirrus_general_file ${base_cirrus_general_path} \
    --annotation_files ${annotation_jsonl_path} \
    --annotation_keys ${annotation_keys} \
    --inject-datestamp base \
    --output_file ${base_all_dataset}
# 全サンプルにラベル付与されたデータセットを作成
show-exec MECABRC=/etc/mecabrc python3 ./shinra-classify/make_dataset.py \
    --cirrus_content_file ${base_cirrus_content_path} \
    --cirrus_general_file ${base_cirrus_general_path} \
    --annotation_files ${annotation_jsonl_path} \
    --annotation_keys ${annotation_keys} \
    --inject-datestamp base \
    --labeled_only \
    --output_file ${base_labeled_dataset}
# 新バージョンのWikipedia記事を元にラベル無しデータセットを作成
show-exec MECABRC=/etc/mecabrc python3 ./shinra-classify/make_dataset.py \
    --cirrus_content_file ${new_cirrus_content_path} \
    --cirrus_general_file ${new_cirrus_general_path} \
    --inject-datestamp new \
    --annotation_keys __UNUSED__ \
    --unlabeled_only \
    --output_file ${new_nolabel_dataset}
# 旧バージョンのアノテーションと新バージョンの全記事データをマージ
show-exec-pv pv -c ${base_labeled_dataset} ${new_nolabel_dataset} \
    \| pv -cl \
    \> ${merged_all_dataset}
# 語彙ファイルを作成 (未知語は処理できないため、マージデータを利用)
show-exec python3 shinra-classify/build_vocab.py \
    --dataset_file ${merged_all_dataset} \
    --output_file ${vocab_file} \
    --feature_vocab_size ${FEATURE_VOCAB_SIZE}
# 訓練
show-exec python3 shinra-classify/train.py \
    --train_file ${base_labeled_dataset} \
    --vocab_file  ${vocab_file} \
    --entity_embed_file ${entity_embed_file} \
    --entity_embed_size ${ENTITY_EMBED_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --epoch ${TRAIN_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --device ${USE_GPU} \
    --output_dir ${training_dir}
## 全データの推論
show-exec python3 shinra-classify/predict.py \
    --dataset_file ${new_nolabel_dataset} \
    --vocab_file ${vocab_file} \
    --entity_embed_size ${ENTITY_EMBED_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --model_file ${model_file} \
    --batch_size ${BATCH_SIZE} \
    --calc_scores 0 \
    --device ${USE_GPU} \
    --output_key ${OUTPUT_KEY} \
    --output_file ${OUTPUT_DIR}/predict-all.jsonl
show-exec python3 shinra-classify/format-for-shinra.py \
    --input_file ${OUTPUT_DIR}/predict-all.tagged.jsonl \
    --tag new \
    --definition ${ene_definition_jsonl_path} \
    \> ${OUTPUT_DIR}/predict-all.jsonl

# n-best推論 (補助目的なので不要ならコメントアウト)
show-exec python3 shinra-classify/predict.py \
    --dataset_file ${new_nolabel_dataset} \
    --vocab_file ${vocab_file} \
    --entity_embed_size ${ENTITY_EMBED_SIZE} \
    --hidden_size ${HIDDEN_SIZE} \
    --model_file ${model_file} \
    --batch_size ${BATCH_SIZE} \
    --calc_scores 0 \
    --device ${USE_GPU} \
    --nbest ${N_BEST} \
    --output_key ${OUTPUT_KEY} \
    --output_file ${OUTPUT_DIR}/predict-all.${N_BEST}-best.jsonl
show-exec python3 shinra-classify/format-for-shinra.py \
    --input_file ${OUTPUT_DIR}/predict-all.${N_BEST}-best.tagged.jsonl \
    --tag new \
    --definition ${ene_definition_jsonl_path} \
    \> ${OUTPUT_DIR}/predict-all.${N_BEST}-best.jsonl

# クロスバリデーション(検証用のため不要ならコメントアウト)
#show-exec python3 shinra-classify/cross_validation.py \
#    --dataset_file ${base_labeled_dataset} \
#    --vocab_file  ${vocab_file} \
#    --entity_embed_file ${entity_embed_file} \
#    --entity_embed_size ${ENTITY_EMBED_SIZE} \
#    --hidden_size ${HIDDEN_SIZE} \
#    --cv_fold ${CV_FOLD} \
#    --epoch ${CV_EPOCHS} \
#    --batch_size ${BATCH_SIZE} \
#    --eval_batch_size ${BATCH_SIZE} \
#    --device ${USE_GPU} \
#    --output_dir ${cross_validation_dir}
