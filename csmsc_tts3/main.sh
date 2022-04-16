#! /bin/bash

inference_dir="$PWD/resources"

am="fastspeech2_csmsc"
am_onnx="fastspeech2_csmsc_onnx_0.2.0/fastspeech2_csmsc.onnx"

voc="hifigan_csmsc"
voc_onnx="hifigan_csmsc.onnx"

output_dir="result"
text="$PWD/csmsc_test.txt"
phones_dict="$PWD/resources/fastspeech2_csmsc_onnx_0.2.0/phone_id_map.txt"

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
