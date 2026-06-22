#!/bin/bash
export MODELSCOPE_CACHE="./pretrained_models"

config_path=./data/config/soulx_transcriber.yaml
model_dir=./pretrained_models/soulx_transcriber
mkdir -p ./pretrained_models
# download the model
if [ ! -d $model_dir ]; then
  echo "download SoulX-Transcriber model weights"
  modelscope download --model Soul-AILab/SoulX-Transcriber --local_dir $model_dir --max-workers 8
fi

# podcast test wav
wav_path=./data/audios/movie.wav
#output dir
out_dir=./data/output
mkdir -p $out_dir

# inference
echo "Inference model: $model_dir" 
echo "Config: $config_path"
echo "Output directory: $out_dir"
echo "WAV path: $wav_path"

export CUDA_VISIBLE_DEVICES="0"
python ./inference/infer.py \
  --model $model_dir \
  --audio-path $wav_path \
  --output-dir $out_dir \
  --stage-configs-path $config_path \
  --temperature 0.2 \
  --top-p 0.9 \
  --top-k -1 \
  --max-tokens 32768 \
