
# TensorRT-LLM Canary Setup and Usage Guide


## Setup Instructions



### 1. Docker Commands
```bash
docker pull appsprodacr.azurecr.io/trtllm_canary:latest
docker run --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --gpus device=0 \
 -v <path to output-dir>:/inference
  -v <path to the dir of trt_engine>:/models \
  appsprodacr.azurecr.io/trtllm_canary:latest
```

### 2. Running inference
```bash
# decode a single wav file
python3 run.py --engine_dir <engine_dir> --nemo_dir <path to nemo checkpoint> --name single_wav_test --batch_size=1 --num_beam=<beam_len> --enable_warmup --input_file assets/1221-135766-0002.wav --results_dir <path>

# decode a whole dataset
python3 run.py --engine_dir <engine_dir> --dataset hf-internal-testing/librispeech_asr_dummy --enable_warmup  --batch_size=<batch_size> --num_beam=<beam_len>  --name librispeech_dummy_large_v3

# decode with a manifest file and save to manifest.
python3 run.py --engine_dir <engine_dir> --enable_warmup --batch_size=<batch_size> --num_beam=<beam_len> --name <test_name> --manifest_file <path_to_manifest_file>

```
