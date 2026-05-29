# Python API

Python API 适合在应用中集成 RapidTTS。核心流程是创建 `RapidTTS`，构造 `SynthesisRequest`，再保存 `SynthesisResponse`。

## 最小示例

```python
from rapidtts import RapidTTS, SynthesisRequest

tts = RapidTTS()

resp = tts.synthesize(
    SynthesisRequest(text="我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。")
)

resp.save("result.wav")
```

未指定模型时，RapidTTS 使用配置中的默认后端，目前是 `kokoro_onnx`。

## 指定模型

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)
resp = tts.synthesize(SynthesisRequest(text="你好，RapidTTS"))
resp.save("kokoro.wav")
```

使用 MeloTTS ONNX：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MELO_ONNX)
resp = tts.synthesize(SynthesisRequest(text="你好，RapidTTS"))
resp.save("melo.wav")
```

使用 MOSS Nano ONNX：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="Junhao",
    )
)
resp.save("moss_nano.wav")
```

可选模型：

- `TTSModel.KOKORO_ONNX`：默认后端，支持多音色
- `TTSModel.MELO_ONNX`：可选后端，当前只暴露 `zf_001` 音色
- `TTSModel.MOSS_NANO_ONNX`：MOSS Nano 后端，支持内置音色和参考音频克隆

## 指定音色

音色统一通过 `SynthesisRequest.voice` 指定，不放在 `extras` 中。

Kokoro ONNX 支持多音色：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)

resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="zm_009",
    )
)
resp.save("kokoro_zm_009.wav")
```

查询当前模型可用音色：

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)
print(tts.get_voices())
```

MeloTTS ONNX 当前只暴露 `zf_001`：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MELO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="zf_001",
    )
)
resp.save("melo_zf_001.wav")
```

MOSS Nano ONNX 的内置音色也通过 `voice` 指定，默认音色是 `Junhao`：

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
resp = tts.synthesize(
    SynthesisRequest(
        text="你好，RapidTTS",
        voice="Ava",
    )
)
resp.save("moss_nano_ava.wav")
```

MOSS Nano 的内置音色列表来自模型 manifest。当前内置音色如下：

```text
Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu, Trump, Ava, Bella, Adam, Nathan, Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

也可以运行时查询，避免本地模型文件更新后和文档不一致：

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)
print(tts.get_voices())
```

## MOSS Nano 参考音频

MOSS Nano ONNX 支持通过参考音频进行音色克隆。参考音频路径放在 `SynthesisRequest.extras["prompt_audio_path"]`：

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

如果同时传入 `voice` 和 `prompt_audio_path`，MOSS Nano 会优先使用参考音频。首次使用参考音频时会自动下载 `prompt_audio_encoder` 可选模型文件组；也可以提前执行：

```bash
rapidtts download moss_nano_onnx --group prompt_audio_encoder
```

`extras["prompt_text"]` 也会被读取并做文本归一化，保留给 MOSS Nano 请求扩展使用。

## 指定语言、语速和采样率

```python
from rapidtts import RapidTTS, SynthesisRequest, TTSLanguage

tts = RapidTTS()

resp = tts.synthesize(
    SynthesisRequest(
        text="hello world",
        language=TTSLanguage.EN,
        speed=1.1,
        sample_rate=16000,
    )
)
resp.save("en.wav")
```

常用语言：

- `TTSLanguage.ZH`：中文
- `TTSLanguage.EN`：英文
- `TTSLanguage.ZH_MIX_EN`：中英混合

注意：MOSS Nano ONNX 当前后处理会按模型输出返回 48000 Hz 音频，不做 `speed` 变速处理；如需变速或重采样，可在保存后用外部音频处理流程完成。

## 指定模型目录

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(
    model=TTSModel.KOKORO_ONNX,
    model_root_dir="/path/to/kokoro_onnx",
)
```

如果不指定 `model_root_dir`，RapidTTS 会使用默认模型目录，并在模型缺失时自动下载。

MOSS Nano 的默认模型目录是 `models/moss_nano_onnx`，自定义目录示例：

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(
    model=TTSModel.MOSS_NANO_ONNX,
    model_root_dir="/path/to/moss_nano_onnx",
)
```

## 查看模型能力

```python
from rapidtts import RapidTTS, TTSModel

tts = RapidTTS(model=TTSModel.KOKORO_ONNX)

capability = tts.get_capability()
print(capability.name)
print(capability.languages)
print(capability.default_language)
print(capability.default_voice)
print(capability.voices)
```

`get_capability()` 返回 `ModelCapability`，包含模型名称、支持语言、默认语言、音色列表、默认音色和音色来源。

## 文本归一化

RapidTTS 默认使用 WeText 文本归一化。也可以显式指定归一化器：

```python
from rapidtts import RapidTTS, TextNormalizerType, TTSModel

tts = RapidTTS(
    model=TTSModel.KOKORO_ONNX,
    text_normalizer_type=TextNormalizerType.WETEXT,
)
```

可选值：

- `TextNormalizerType.WETEXT`
- `TextNormalizerType.LEGACY`
- `TextNormalizerType.NONE`

## 请求字段

`SynthesisRequest` 常用字段如下：

|字段|说明|
|:---|:---|
|`text`|待合成文本|
|`language`|语言，可选 `ZH`、`EN`、`ZH_MIX_EN`|
|`voice`|音色名称，例如 `zf_001`、`zm_009`|
|`speed`|语速，默认 `1.0`|
|`sample_rate`|输出采样率|
|`audio_format`|音频格式，默认 `wav`|
|`extras`|后端扩展参数，不用于传递 `voice`；MOSS Nano 可使用 `prompt_audio_path`、`prompt_text`|
