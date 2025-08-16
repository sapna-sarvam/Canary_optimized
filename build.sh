# install requirements first
#pip install -r requirements.txt

INFERENCE_PRECISION=bfloat16 # precision float16 or bfloat16
MAX_BEAM_WIDTH=4 # max beam width of decoder
MAX_BATCH_SIZE=96 # max batch size
MAX_FEAT_LEN=6001 #Max audio duration(ms)/10ms (window shift). Assuming 30s audio
MAX_ENCODER_OUTPUT_LEN=400 #MAX_ENCODER_OUTPUT_LEN = 1 + (MAX_FEAT_LEN / 8), 8 is subsampling factor for canary conformer
MAX_TOKENS=196 # Max number of tokens to generate
MAX_PROMPT_TOKENS=10 # Max number of tokens to be passed


engine_dir="engine"_${INFERENCE_PRECISION}
checkpoint_dir="tllm_checkpoint"_${INFERENCE_PRECISION}
NEMO_MODEL="nvidia/canary-1b-flash"
MODEL_PATH="/inference/examples/canary/canary_model/canary_in22_multilingual_tokenizer_iv_ft-averaged.nemo"


# Export the canary model TensorRT-LLM format.
python3 convert_checkpoint.py \
                --dtype=${INFERENCE_PRECISION} \
                --model_name ${NEMO_MODEL} \
                --output_dir ${checkpoint_dir} \
                --model_path ${MODEL_PATH} \
                ${engine_dir}


# Build the canary encoder model using conformer_onnx_trt.py
python3 conformer_onnx_trt.py \
        --max_BS ${MAX_BATCH_SIZE} \
        --max_feat_len ${MAX_FEAT_LEN} \
        ${checkpoint_dir} \
        ${engine_dir}


# Build the canary decoder  using trtllm-build
trtllm-build  --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${engine_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len ${MAX_TOKENS} \
              --max_input_len ${MAX_PROMPT_TOKENS} \
              --max_encoder_input_len ${MAX_ENCODER_OUTPUT_LEN}  \
             --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin disable \
              --gpt_attention_plugin ${INFERENCE_PRECISION} \
              --remove_input_padding enable
