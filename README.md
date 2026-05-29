<div align="center">
  <img src="https://github.com/RapidAI/RapidTTS/releases/download/v1.0.1/logo.jpg" width="80%">

  <h1><b>🎙 RapidTTS</b></h1>
  <b><font size="4"><i> 轻量级文本转语音工具，面向本地快速推理 </i></font></b>
  <div>&nbsp;</div>

<a href="https://www.modelscope.cn/studios/RapidAI/RapidTTS/summary" target="_blank"><img src="https://img.shields.io/badge/ModelScope-Demo-blue?style=flat-square"></a>
<a href=""><img alt="Python" src="https://img.shields.io/badge/Python-%3E%3D3.9%2C%3C4-aff.svg?style=flat-square&logo=python&logoColor=white"></a>
<a href=""><img alt="OS" src="https://img.shields.io/badge/OS-Linux%2C%20Win%2C%20Mac-pink.svg?style=flat-square"></a>
<a href="https://pypi.org/project/rapidtts/"><img alt="PyPI" src="https://img.shields.io/pypi/v/rapidtts?style=flat-square&logo=pypi&logoColor=white&color=3775A9"></a>
<a href="https://pepy.tech/project/rapidtts"><img src="https://static.pepy.tech/personalized-badge/rapidtts?style=flat-square&period=total&units=abbreviation&left_color=grey&right_color=blue&left_text=Downloads"></a>
<a href="https://semver.org/"><img alt="SemVer" src="https://img.shields.io/badge/semver-2.0-3D9970?style=flat-square"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000?style=flat-square"></a>
</div>

## 简介

RapidTTS 是一个轻量级文本转语音工具，面向本地快速推理。当前默认后端是 `kokoro_onnx`，同时支持 `melo_onnx` 和 `moss_nano_onnx`。

## 特性

- 支持 Kokoro ONNX、MeloTTS ONNX 和 MOSS Nano ONNX 推理
- 支持中文、英文和中英混合文本
- 支持查询模型语言、默认参数和音色能力
- 模型文件可自动下载，并使用 SHA256 校验
- 同时提供 Python API 和命令行工具

## [在线 demo](https://www.modelscope.cn/studios/RapidAI/RapidTTS/summary) 

<a href="https://www.modelscope.cn/studios/RapidAI/RapidTTS/summary" target="_blank"><img src="https://github.com/RapidAI/RapidTTS/releases/download/v1.2.0/online_demo.jpg"></a>

## 安装

默认推荐安装 Kokoro ONNX 后端：

```bash
pip install "rapidtts[kokoro]"
```

使用 MOSS Nano ONNX 时安装对应 extra：

```bash
pip install "rapidtts[moss_nano]"
```

其他安装方式见 [安装说明](docs/installation.md)。

## 快速使用

### Python API

```python
from rapidtts import RapidTTS, SynthesisRequest

tts = RapidTTS()
resp = tts.synthesize(SynthesisRequest(text="你好，RapidTTS"))
resp.save("outputs/1.wav")
```

指定模型和音色：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="zm_009",
    )
)
resp.save("outputs/zm_009.wav")
```

使用 MOSS Nano 内置音色：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="Junhao",
    )
)
resp.save("outputs/moss_nano_junhao.wav")
```

当前 MOSS Nano 内置音色：

```text
Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu, Trump, Ava, Bella, Adam, Nathan, Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

MOSS Nano 也支持通过 `extras["prompt_audio_path"]` 传入参考音频。首次使用参考音频时会自动下载 `prompt_audio_encoder` 可选模型文件组。

更多示例见 [Python API](docs/python_api.md)。

### 命令行

```bash
rapidtts text "你好，RapidTTS" outputs/1.wav
```

指定模型：

```bash
rapidtts text "你好，RapidTTS" outputs/kokoro.wav --model kokoro_onnx
rapidtts text "你好，RapidTTS" outputs/moss_nano.wav --model moss_nano_onnx --voice Junhao
```

指定模型和音色：

```bash
rapidtts text "你好，RapidTTS" outputs/zm_009.wav --model kokoro_onnx --voice zm_009
```

更多命令见 [命令行用法](docs/cli.md)。

## 文档

- [安装说明](docs/installation.md)：安装 extra、检查依赖和模型文件
- [Python API](docs/python_api.md)：在代码中指定模型、音色、语言、语速和模型目录
- [命令行用法](docs/cli.md)：下载模型、检查安装、查询能力、合成音频
- [模型信息](docs/models.md)：支持的模型、语言、音色规则和模型文件
- [开发说明](docs/development.md)：测试、文本归一化和项目结构

## 致谢

- [kokoro](https://github.com/hexgrad/kokoro)
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- [MeloTTS](https://github.com/myshell-ai/MeloTTS)
- [melotts-onnx](https://pypi.org/project/melotts-onnx/)
- MOSS-TTS-Nano
- [wetext](https://github.com/pengzhendong/wetext)

## Star history

[![Stargazers over time](https://starchart.cc/RapidAI/RapidTTS.svg?variant=adaptive)](https://starchart.cc/RapidAI/RapidTTS)

## License

Apache-2.0 License。
