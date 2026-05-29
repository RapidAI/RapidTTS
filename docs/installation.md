# 安装说明

默认推荐安装 Kokoro ONNX 后端：

```bash
pip install "rapidtts[kokoro]"
```

## 安装选项

```bash
pip install rapidtts            # 只安装核心依赖
pip install "rapidtts[kokoro]"  # 安装默认 Kokoro ONNX 后端
pip install "rapidtts[melo]"    # 安装 MeloTTS ONNX 后端
pip install "rapidtts[moss_nano]" # 安装 MOSS Nano ONNX 后端
pip install "rapidtts[all]"     # 安装全部后端依赖
```

只安装 `rapidtts` 时不会安装具体后端的额外依赖。运行某个后端前，请安装对应 extra。

## 检查安装

安装完成后运行：

```bash
rapidtts check
```

该命令会检查：

- RapidTTS 包是否可以正常导入
- 当前模型所需依赖是否存在
- `config.yaml` 和 `models.yaml` 是否可解析
- 模型文件是否存在，SHA256 是否正确

如果还希望验证后端能否初始化：

```bash
rapidtts check --init-backend
```

## 下载模型

如果模型文件缺失，可以手动下载：

```bash
rapidtts download kokoro_onnx
```

下载到自定义目录：

```bash
rapidtts download kokoro_onnx --save-dir /path/to/kokoro_onnx
```

检查自定义目录：

```bash
rapidtts check kokoro_onnx --model-dir /path/to/kokoro_onnx
```

## MeloTTS ONNX

使用 MeloTTS ONNX 时安装对应 extra：

```bash
pip install "rapidtts[melo]"
rapidtts check melo_onnx
```

## MOSS Nano ONNX

使用 MOSS Nano ONNX 时安装对应 extra：

```bash
pip install "rapidtts[moss_nano]"
rapidtts check moss_nano_onnx
```

MOSS Nano 的基础模型文件支持内置音色合成。使用 `SynthesisRequest.extras["prompt_audio_path"]` 传入自定义参考音频时，还需要 `prompt_audio_encoder` 可选模型文件组；运行合成时会按需自动下载，也可以提前下载：

```bash
rapidtts download moss_nano_onnx --group prompt_audio_encoder
```

如果基础模型文件已经存在，只想检查或下载参考音频编码器：

```bash
rapidtts check moss_nano_onnx --group prompt_audio_encoder --no-base-files
rapidtts download moss_nano_onnx --group prompt_audio_encoder --no-base-files
```
