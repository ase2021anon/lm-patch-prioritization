#!/bin/bash

CLASS=$1 # defects4j project name (e.g. Chart)
BUG_NUM=$2 # defects4j bug number (e.g. 1)
TOOL_NAME="TBar" # APR tool name
NAT_METHOD="Avg" # naturalness calculation method (this only functions as description)
TICKER="HM-${NAT_METHOD}" # additional description string
PREPROCESSING=0 # if 1, will run TBar tool to get patches
TBAR_DIR="~/Documents/other-papers/TBar" # TBar repository location
XML_DIR="${TBAR_DIR}/OUTPUT/PerfectFL/TBar/AllPatches/${CLASS}_${BUG_NUM}" # location of resulting patches
CURR_DIR=$(pwd)
TMP_DIR="${CURR_DIR}/data/" # temporary directory to save data
BPE_OPS_FILE="${CURR_DIR}/tokenizing/bpe_data/jm-trainfunc_varpairs_BPE.pkl" # saved BPE file
EXPR_PREFIX="${TOOL_NAME}${CLASS}${BUG_NUM}" # description string
RESULT_DIR="${CURR_DIR}/nat_results" # directory where naturalness results are saved as csv file
MODEL_PATH="./bpe_lm/models/weights/SrcMLBPE_JMJavaFunc_LMl1_z1000.pth" # trained language model path
FINETUNED=0 # whether to use overfit models or not (not in paper)

echo $XML_DIR

## Step 0. Get all the patches by executing TBar.
if [ $PREPROCESSING -eq 1 ]
then
    cd $TBAR_DIR
    PATCH_NAME="${CLASS}_${BUG_NUM}"
    sh patch_processing.sh $PATCH_NAME
    cd $CURR_DIR
    exit 0
fi

## Step 1. Change XML files to tokenized lines and leave only target functions.
XML_TOK_FILE="${TMP_DIR}${EXPR_PREFIX}tokenized.txt"
cd ..
python -m bpe_lm.tokenizing.tokenize_xml --source_dir $XML_DIR --target_file $XML_TOK_FILE --len_threshold 3000
cd $CURR_DIR

## Step 2. BPE parsing.
BPE_PKL_FILE="${TMP_DIR}${EXPR_PREFIX}bpe.pkl"
cd tokenizing
python bpe_parsing.py --BPE_ops_file $BPE_OPS_FILE --tokenized_file $XML_TOK_FILE --target_file $BPE_PKL_FILE
cd ..

## Step 3. Calculating Naturalness
RESULT_EXPR_PREFIX="${TOOL_NAME}-${TICKER}${CLASS}${BUG_NUM}"
OUTPUT_FILE="${RESULT_DIR}/${EXPR_PREFIX}.${NAT_METHOD}-nfl-jmtrain.csv"
cd ..
python -m bpe_lm.calc_naturalness --eval_file $BPE_PKL_FILE --ref_dir $XML_DIR --only_plausible 1 --visualization_file "bpe_lm/visualizations/${RESULT_EXPR_PREFIX}.html" --bug_proj $CLASS --bug_num $BUG_NUM --model_path $MODEL_PATH --finetuned $FINETUNED --use_suspiciousness 0 --nat_csv $OUTPUT_FILE
cd $CURR_DIR
