# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

model_root_dir = "rapidtts/models/moss_nano_onnx"
tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX, model_root_dir=model_root_dir)

text = "2026年5月8日，猪肉价格为每斤13.5元，较前期上涨6.3%；车牌号码为京A86F29。"
ref_audio_path = "rapidtts/models/moss_nano_onnx/assets/audio/zh_6.wav"
result = tts.synthesize(
    SynthesisRequest(text=text, extras={"prompt_audio_path": ref_audio_path})
)

save_path = "outputs/moss_nano_result.wav"
result.save(save_path)
print(save_path)
