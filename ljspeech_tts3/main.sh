#! /bin/bash

inference_dir="$PWD/resources"

am="fastspeech2_ljspeech"
am_onnx="fastspeech2_ljspeech/fastspeech2_ljspeech.onnx"

voc="pwgan_ljspeech"
voc_onnx="pwgan_ljspeech.onnx"

output_dir="result"
text="$PWD/sentences_en.txt"
phones_dict="$PWD/resources/fastspeech2_ljspeech/phone_id_map.txt"

python tts3.py \
    --inference_dir=${inference_dir} \
    --am=${am} \
    --am_onnx=${am_onnx} \
    --voc=${voc} \
    --voc_onnx=${voc_onnx} \
    --output_dir=${output_dir} \
    --text=${text} \
    --phones_dict=${phones_dict} \
    --device=cpu \
    --cpu_threads=2
