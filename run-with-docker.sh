#!/bin/bash

set -eu

script_dir="$(cd "$(dirname "${BASH_SOURCE:-${(%):-%N}}")"; pwd)"                                                       
source ${script_dir}/common.sh 

# 出力ディレクトリ
OUTPUT_DIR=${OUTPUT_DIR:-output}

# Dockerイメージ名
DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-shinra-classify}

# 並列処理設定
NUM_WORKERS=${NUM_WORKERS:-2}

# タイムゾーン (ログの日時を正しく取得するため)
TZ=${TZ:-Asia/Tokyo}

# GPU設定 (GPU番号、0オリジン、-1以下ならGPU不使用)
USE_GPU=${USE_GPU:-0}

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

check-file ${base_cirrus_content_path}
check-file ${new_cirrus_content_path}
check-file ${base_cirrus_general_path}
check-file ${new_cirrus_general_path}
check-file ${annotation_jsonl_path}
check-file ${ene_definition_jsonl_path}

tmp_base_cirrus_content_path=/tmp/$(basename ${base_cirrus_content_path})
tmp_new_cirrus_content_path=/tmp/$(basename ${new_cirrus_content_path})
tmp_base_cirrus_general_path=/tmp/$(basename ${base_cirrus_general_path})
tmp_new_cirrus_general_path=/tmp/$(basename ${new_cirrus_general_path})
tmp_annotation_jsonl_path=/tmp/$(basename ${annotation_jsonl_path})
tmp_ene_definition_jsonl_path=/tmp/$(basename ${ene_definition_jsonl_path})

DOCKER_RUN_OPTIONS=""
if [ "${USE_GPU}" -a "${USE_GPU}" -ge 0 ]; then
    DOCKER_RUN_OPTIONS="--runtime=nvidia"
fi

#show-exec docker build -t ${DOCKER_IMAGE_NAME} .
show-exec docker run --rm -it \
    -v $(readlink -f ${OUTPUT_DIR}):/app/output \
    -v $(readlink -f ${base_cirrus_content_path}):${tmp_base_cirrus_content_path} \
    -v $(readlink -f ${new_cirrus_content_path}):${tmp_new_cirrus_content_path} \
    -v $(readlink -f ${base_cirrus_general_path}):${tmp_base_cirrus_general_path} \
    -v $(readlink -f ${new_cirrus_general_path}):${tmp_new_cirrus_general_path} \
    -v $(readlink -f ${annotation_jsonl_path}):${tmp_annotation_jsonl_path} \
    -v $(readlink -f ${ene_definition_jsonl_path}):${tmp_ene_definition_jsonl_path} \
    -v ${PWD}/run.sh:/app/run.sh \
    -v ${PWD}/common.sh:/app/common.sh \
    -v ${PWD}/shinra-classify:/app/shinra-classify \
    -e NUM_WORKERS=${NUM_WORKERS} \
    -e OUTPUT_DIR=/app/output \
    -e TZ=${TZ} \
    -e USE_GPU=${USE_GPU} \
    ${DOCKER_RUN_OPTIONS} \
    ${DOCKER_IMAGE_NAME} \
    ${tmp_base_cirrus_content_path} \
    ${tmp_new_cirrus_content_path} \
    ${tmp_base_cirrus_general_path} \
    ${tmp_new_cirrus_general_path} \
    ${tmp_annotation_jsonl_path} \
    ${tmp_ene_definition_jsonl_path}
