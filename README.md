#### 说明
- 基于PaddleSpeech下的TTS2整理而来
- 共分为三步，`frontend`、`acoustic`、`vocoder`
  - 其中`acoustic`这一步模型推理目前基于`PaddlePaddle`,
  - `vocoder`模型推理基于ONNXRuntime

#### 运行步骤
1. 安装`requirements.txt`
   ```bash
   pip install -r requirements.txt -i https://pypi.douban.com/simple/
   ```

2. 运行`tts2.py`
   ```bash
   python tts2.py
   ```
3. 运行日志如下:
   ```text
   初始化前处理部分
    frontend done!
    初始化提取特征模型
    am_predictor done!
    初始化合成wav模型
    合成指定句子
    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 1.431 seconds.
    Prefix dict has been built successfully.
    infer_result/001.wav done!      cost: 7.226019859313965s
    infer_result/002.wav done!      cost: 9.149477005004883s
    infer_result/003.wav done!      cost: 3.4020116329193115s
    infer_result/004.wav done!      cost: 14.5472412109375s
    infer_result/005.wav done!      cost: 14.142913818359375s
    infer_result/006.wav done!      cost: 10.191686630249023s
    infer_result/007.wav done!      cost: 15.726643800735474s
    infer_result/008.wav done!      cost: 15.421608209609985s
    infer_result/009.wav done!      cost: 8.083441972732544s
    infer_result/010.wav done!      cost: 10.538750886917114s
    infer_result/011.wav done!      cost: 7.974739074707031s
    infer_result/012.wav done!      cost: 7.274432897567749s
    infer_result/013.wav done!      cost: 8.204563856124878s
    infer_result/014.wav done!      cost: 8.994312286376953s
    infer_result/015.wav done!      cost: 5.084768056869507s
    infer_result/016.wav done!      cost: 5.3102569580078125s
   ```