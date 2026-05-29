# 模型信息

RapidTTS 的模型选择由 `TTSModel` 或 CLI 的 `MODEL` 参数控制。当前支持 `kokoro_onnx`、`melo_onnx` 和 `moss_nano_onnx`。

模型托管在：[魔搭平台](https://www.modelscope.cn/models/RapidAI/RapidTTS/tree/master/models)

## 支持矩阵

|模型|模型存储大小|语言|音色|默认采样率|说明|
|:---|:---|:---|:---|:---|:---|
|`kokoro_onnx`|389M|`ZH_MIX_EN`|多音色，默认 `zf_001`|`24000`|默认后端|
|`melo_onnx`|752M|`ZH_MIX_EN`|单音色，仅支持 `zf_001`|`44100`|需要安装 `rapidtts[melo]`|
|`moss_nano_onnx`|约 685M，参考音频编码器约 43M|`ZH_MIX_EN`|内置多音色，默认 `Junhao`；支持参考音频克隆|`48000`|需要安装 `rapidtts[moss_nano]`|

MOSS Nano ONNX 当前按模型输出返回 48000 Hz 音频，后处理不做语速变换。

## 查看权威信息

模型能力以运行时查询为准：

```bash
rapidtts info kokoro_onnx
rapidtts voices kokoro_onnx
rapidtts info moss_nano_onnx
rapidtts voices moss_nano_onnx
```

Python 中也可以查询：

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)
capability = tts.get_capability()

print(capability.languages)
print(capability.default_voice)
print(capability.voices)
```

## 音色命名

Kokoro ONNX 的音色名称通常带有前缀：

|前缀|含义|
|:---|:---|
|`zf`|Chinese female，中文女声|
|`zm`|Chinese male，中文男声|
|`af`|American female，美式英语女声|
|`bf`|British female，英式英语女声|

示例：

- `zf_001`：默认中文女声
- `zm_009`：中文男声
- `af_maple`：美式英语女声
- `bf_vale`：英式英语女声

完整列表请用 `rapidtts voices kokoro_onnx` 查询，避免文档中的静态列表和模型文件不一致。

CLI 合成时可以通过 `--voice` 指定音色：

```bash
rapidtts text "你好，RapidTTS" outputs/zm_009.wav --model kokoro_onnx --voice zm_009
```

Python API 中使用 `SynthesisRequest.voice` 指定音色。`extras` 只用于后端扩展参数，不用于传递音色。

## MOSS Nano 音色和参考音频

MOSS Nano ONNX 的内置音色来自 `MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json`。当前 manifest 中包含这些音色：

|序号|音色|
|:---|:---|
|1|`Junhao`|
|2|`Zhiming`|
|3|`Weiguo`|
|4|`Xiaoyu`|
|5|`Yuewen`|
|6|`Lingyu`|
|7|`Trump`|
|8|`Ava`|
|9|`Bella`|
|10|`Adam`|
|11|`Nathan`|
|12|`Soyo`|
|13|`Saki`|
|14|`Mortis`|
|15|`Umiri`|
|16|`Mei`|
|17|`Anon`|
|18|`Arisa`|

完整列表仍建议用运行时命令查询：

```bash
rapidtts voices moss_nano_onnx
```

使用内置音色时，通过 `voice` 传入音色名称：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="Junhao",
    )
)
resp.save("moss_nano_junhao.wav")
```

使用参考音频克隆时，通过 `extras["prompt_audio_path"]` 传入音频路径。此时 `voice` 会被参考音频覆盖：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        extras={"prompt_audio_path": "/path/to/reference.wav"},
    )
)
resp.save("moss_nano_clone.wav")
```

参考音频会被重采样到 MOSS Audio Tokenizer 需要的采样率和声道数。首次使用参考音频时，RapidTTS 会自动下载 `prompt_audio_encoder` 可选模型文件组；也可以提前下载：

```bash
rapidtts download moss_nano_onnx --group prompt_audio_encoder
```

## 模型文件

模型文件托管在 ModelScope：

```text
https://www.modelscope.cn/models/RapidAI/RapidTTS/tree/master/models
```

模型元数据声明在：

```text
rapidtts/configs/models.yaml
```

其中包含：

- 下载地址 `base_url`
- 默认本地目录 `local_dir`
- 文件列表
- 每个文件的 SHA256

## 默认模型目录

如果没有指定模型目录，RapidTTS 会按 `models.yaml` 中的 `local_dir` 使用默认目录，并在模型缺失时自动下载。

默认目录相对于 `rapidtts` 包目录：

```text
<rapidtts package root>/models/kokoro_onnx
<rapidtts package root>/models/melotts_zh_mix_en_onnx
<rapidtts package root>/models/moss_nano_onnx
```

Python 中指定目录：

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(
    model=TTSModel.KOKORO_ONNX,
    model_root_dir="/path/to/kokoro_onnx",
)
```

CLI 中指定目录：

```bash
rapidtts text "你好，RapidTTS" outputs/1.wav --model kokoro_onnx --model-dir /path/to/kokoro_onnx
```

## 下载和校验

下载模型：

```bash
rapidtts download kokoro_onnx
rapidtts download moss_nano_onnx
```

下载到自定义目录：

```bash
rapidtts download kokoro_onnx --save-dir /path/to/kokoro_onnx
rapidtts download moss_nano_onnx --save-dir /path/to/moss_nano_onnx
```

检查模型文件：

```bash
rapidtts check kokoro_onnx
rapidtts check moss_nano_onnx
```

每个模型文件下载完成后都会计算 SHA256，并与 `models.yaml` 中声明的值比较。本地文件存在且 SHA256 正确时会直接复用；文件缺失或 SHA256 不一致时会重新下载。
