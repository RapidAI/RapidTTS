# 命令行用法

RapidTTS 安装后会提供 `rapidtts` 命令。CLI 适合下载模型、检查环境、查看模型能力，以及从文本生成音频文件。

## 命令速查

```bash
rapidtts download [MODEL] [--save-dir DIR] [--group GROUP] [--no-base-files] [--no-progress] [--quiet]
rapidtts check [MODEL] [--model-dir DIR] [--group GROUP] [--no-base-files] [--init-backend] [--quiet]
rapidtts info [MODEL] [--model-dir DIR] [--quiet]
rapidtts voices [MODEL] [--model-dir DIR] [--quiet]
rapidtts text TEXT OUTPUT [--model MODEL] [--model-dir DIR] [--language ZH|EN|ZH_MIX_EN] [--voice VOICE] [--speed SPEED] [--sample-rate SAMPLE_RATE] [--quiet]
```

`MODEL` 可选：

- `kokoro_onnx`：默认后端
- `melo_onnx`：MeloTTS ONNX 后端
- `moss_nano_onnx`：MOSS Nano ONNX 后端

## 生成音频

最小示例：

```bash
rapidtts text "你好，RapidTTS" outputs/1.wav
```

指定模型：

```bash
rapidtts text "你好，RapidTTS" outputs/kokoro.wav --model kokoro_onnx
rapidtts text "你好，RapidTTS" outputs/melo.wav --model melo_onnx
rapidtts text "你好，RapidTTS" outputs/moss_nano.wav --model moss_nano_onnx
```

指定语言：

```bash
rapidtts text "hello world" outputs/en.wav --language EN
rapidtts text "你好，RapidTTS" outputs/zh.wav --language ZH
rapidtts text "我在学习machine learning" outputs/mix.wav --language ZH_MIX_EN
```

指定音色：

```bash
rapidtts voices kokoro_onnx
rapidtts text "你好，RapidTTS" outputs/zm_009.wav --model kokoro_onnx --voice zm_009
```

`--voice` 对应 Python API 中的 `SynthesisRequest.voice`。

MeloTTS ONNX 当前只暴露 `zf_001`：

```bash
rapidtts text "你好，RapidTTS" outputs/melo_zf_001.wav --model melo_onnx --voice zf_001
```

MOSS Nano ONNX 的内置音色来自模型 manifest，默认音色是 `Junhao`：

```text
Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu, Trump, Ava, Bella, Adam, Nathan, Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
```

```bash
rapidtts voices moss_nano_onnx
rapidtts text "你好，RapidTTS" outputs/moss_nano_ava.wav --model moss_nano_onnx --voice Ava
```

当前 CLI 的 `text` 子命令支持 MOSS Nano 内置音色合成；参考音频克隆需要使用 Python API 的 `SynthesisRequest.extras["prompt_audio_path"]`。

指定语速和采样率：

```bash
rapidtts text "你好，RapidTTS" outputs/fast.wav --speed 1.2 --sample-rate 16000
```

注意：MOSS Nano ONNX 当前按模型输出返回 48000 Hz 音频，不做 `--speed` 变速处理。

使用自定义模型目录：

```bash
rapidtts text "你好，RapidTTS" outputs/1.wav --model kokoro_onnx --model-dir /path/to/kokoro_onnx
rapidtts text "你好，RapidTTS" outputs/moss_nano.wav --model moss_nano_onnx --model-dir /path/to/moss_nano_onnx
```

## 查看模型能力

查看模型支持的语言、默认语言、音色数量和默认音色：

```bash
rapidtts info kokoro_onnx
```

查看模型可用音色：

```bash
rapidtts voices kokoro_onnx
```

查看 MeloTTS ONNX：

```bash
rapidtts info melo_onnx
rapidtts voices melo_onnx
```

查看 MOSS Nano ONNX：

```bash
rapidtts info moss_nano_onnx
rapidtts voices moss_nano_onnx
```

## 下载模型

下载默认模型：

```bash
rapidtts download kokoro_onnx
```

下载 MOSS Nano ONNX：

```bash
rapidtts download moss_nano_onnx
```

下载到自定义目录：

```bash
rapidtts download kokoro_onnx --save-dir /path/to/kokoro_onnx
rapidtts download moss_nano_onnx --save-dir /path/to/moss_nano_onnx
```

关闭进度条：

```bash
rapidtts download kokoro_onnx --no-progress
```

下载可选模型文件组：

```bash
rapidtts download moss_nano_onnx --group prompt_audio_encoder
```

`prompt_audio_encoder` 是 MOSS Nano 参考音频克隆所需的可选文件组。只使用内置音色时不需要下载这个文件组；Python API 首次使用 `extras["prompt_audio_path"]` 时也会按需自动下载。

只下载可选模型文件组，不重复处理基础模型文件：

```bash
rapidtts download moss_nano_onnx --group prompt_audio_encoder --no-base-files
```

## 检查安装

检查默认模型和依赖：

```bash
rapidtts check
```

检查指定模型：

```bash
rapidtts check melo_onnx
rapidtts check moss_nano_onnx
```

检查自定义模型目录：

```bash
rapidtts check kokoro_onnx --model-dir /path/to/kokoro_onnx
```

同时初始化后端：

```bash
rapidtts check kokoro_onnx --init-backend
```

检查可选模型文件组：

```bash
rapidtts check moss_nano_onnx --group prompt_audio_encoder --no-base-files
```

## 关闭日志

所有子命令都支持 `--quiet`：

```bash
rapidtts text "你好，RapidTTS" outputs/1.wav --quiet
```
