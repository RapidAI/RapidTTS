#### RapidTTS2
- **支持合成语言**: 中文和数字(`其他语言后续有空会搞搞`)
- 基于[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)下的[TTS2](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README_cn.md)整理而来
- 共分为三步，`frontend`、`acoustic`、`vocoder`
  - 其中`acoustic`这一步模型推理目前基于`PaddlePaddle`,
  - `vocoder`模型推理基于`ONNXRuntime`
- 其中PaddleSpeech中提供的预训练模型可以参见[link](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README_cn.md#4-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)。在RapidTTS2中使用的是:

    |主要部分|具体模型|支持语言|
    |:---|:---|:---|
    |声学模型|speedyspeech_csmsc|zh|
    |声码器|pwgan_csmsc|zh|

#### 更新日志
##### 🎨2022-04-08 update
- 尝试采用OpenVINO推理引擎，但是目前模型尚未转换成功，具体尝试过程参见:[Paddle模型尝试转换](https://github.com/RapidAI/RapidTTS2/wiki/Paddle%E6%A8%A1%E5%9E%8B%E5%B0%9D%E8%AF%95%E8%BD%AC%E6%8D%A2)

#### 运行步骤
1. 下载`resources`, [Google Drive](https://drive.google.com/file/d/1q3NCydNhFeU2cpLUgevidCHeSzclK0a7/view?usp=sharing) | [百度网盘,提取码:kmcf](https://pan.baidu.com/s/1MGbaS6e_pFqrfIc5OVjWjg), 解压到RapidTTS2目录下

2. 安装`requirements.txt`
   ```bash
   pip install -r requirements.txt -i https://pypi.douban.com/simple/
   ```

3. 运行`tts2.py`
   ```bash
   python tts2.py
   ```
4. 运行日志如下:
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
