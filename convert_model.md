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

#### 转换pwgan_ljspeech和fastspeech_ljspeech到onnx
- pwgan_ljspeech官方只提供了[动态图模型](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc1#pretrained-model)。不过运行代码中提供了动态图转静态图模型代码，只需要搭建PaddleSpeech运行环境，跑一遍示例demo，即可得到对应的静态模型
- 详情参见[AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3359986?shared=1)