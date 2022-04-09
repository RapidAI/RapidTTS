#### 转换pwgan_csmsc到onnx
- `Paddle2ONXX`: latest
- 模型下载: [Pretrained model](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/csmsc/voc1#pretrained-models) | [link](https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_static_0.4.zip)
- 转换脚本:
    ```bash
    paddle2onnx --model_dir pwg_baker_static_0.4 \
                --model_filename pwgan_csmsc.pdmodel \
                --params_filename pwgan_csmsc.pdiparams \
                --save_file pwgan_csmsc.onnx \
                --opset_version 11
    ```
