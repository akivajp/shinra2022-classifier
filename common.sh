#!/bin/bash

# ロギング処理などを行う

OUTPUT_DIR=${OUTPUT_DIR:-output}
LOG_FILE=${LOG_FILE:-run.log}

log_path=${OUTPUT_DIR}/${LOG_FILE}
log_dir=$(dirname ${log_path})

#echo "OUTPUT_DIR: $OUTPUT_DIR"

if [ ! -d ${log_dir} ]; then
    mkdir -p ${log_dir}
fi

red="\033[31m"
yellow="\033[33m"
cyan="\033[36m"
clear="\033[0m"
log-stdout() {
    #format="strftime(\"[stdout at %Y/%m/%d %H:%M:%S]\"), \$0"
    #awk "{print \"${cyan}\"${format}\"${clear}\"}" | tee -a ${log_path}
    local format="\"${cyan}\", strftime(\"[stdeout at %Y/%m/%d %H:%M:%S]\") \"${clear}\", \$0"
    awk "{ print $format }" | tee -a ${log_path}
}
log-stderr() {
    #format="strftime(\"[stdout at %Y/%m/%d %H:%M:%S]\"), \$0"
    #awk "{print \"${red}\"${format}\"${clear}\"}" | tee -a ${log_path} > /dev/stderr
    local format="\"${red}\", strftime(\"[stderr at %Y/%m/%d %H:%M:%S]\") \"${clear}\", \$0"
    awk "{ print $format }" | tee -a ${log_path} > /dev/stderr
}
log-exec() {
    set -u
    local timestamp=$(date "+%Y/%m/%d %H:%M:%S")
    local message="[exec at ${timestamp}] $@"
    echo -e "${yellow}${message}${clear}" | tee -a ${log_path}
    #eval "$@" 2> >(log-stderr) 1> >(log-stdout)
    # NOTE: 失敗した場合、syncしておかないとdockerでログファイル処理されずに終了する可能性あり
    eval "$@" 2> >(log-stderr) 1> >(log-stdout) || (ret=$? ; sync ; return ${ret})
}
show-exec() {
    set -u
    local timestamp=$(date "+%Y/%m/%d %H:%M:%S")
    local message="[exec at ${timestamp}] $@"
    echo -e "${yellow}${message}${clear}" | tee -a ${log_path}
    #eval "$@"
    eval "$@" 2> /dev/stdout | tee -a ${log_path} || (ret=$? ; sync ; return ${ret})
}
show-exec-pv() {
    set -u
    local timestamp=$(date "+%Y/%m/%d %H:%M:%S")
    local message="[exec at ${timestamp}] $@"
    echo -e "${yellow}${message}${clear}" | tee -a ${log_path}
    # NOTE: 1行開けないと表示崩れする場合がある
    echo
    sync
    eval "$@" || (ret=$? ; sync ; return ${ret})
}
show-info() {
    local timestamp=$(date "+%Y/%m/%d %H:%M:%S")
    local message="[info at ${timestamp}] $@"
    echo -e "${cyan}${message}${clear}" | tee -a ${log_path}
}
show-error() {
    local timestamp=$(date "+%Y/%m/%d %H:%M:%S")
    local message="[error at ${timestamp}] $@"
    echo -e "${red}${message}${clear}" | tee -a ${log_path}
}

# ファイル存在チェック
check-file() {
    local path=$1
    if [ ! -f ${path} ]; then
        show-error "File not found: ${path}"
        exit 1
    fi
}