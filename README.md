## RapidTTS(文本转语音)

|目录名称|推理引擎|支持语言|
|:---:|:---:|:---:|
|`csmsc_tts2`|Paddle+ONNXRuntime|中文和数字|
|`csmsc_tts3`|ONNXRuntime|中文和数字|
|`ljspeech_tts3`|ONNXRuntime|中文和数字|

### 更新日志

#### 🎈2022-04-16 update
- 添加`ljspeech_tts3`，英文文本转语音模块

#### 2022-04-09 update
- 添加`csmsc_tts2`中模型转换说明文档([模型转换](./convert_model.md))

#### 2022-04-08 update
- 尝试采用OpenVINO推理引擎，但是目前模型尚未转换成功，具体尝试过程参见:[Paddle模型尝试转换](https://github.com/RapidAI/RapidTTS2/wiki/Paddle%E6%A8%A1%E5%9E%8B%E5%B0%9D%E8%AF%95%E8%BD%AC%E6%8D%A2)
