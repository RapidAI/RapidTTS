# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from rapidtts import RapidTTS, SynthesisRequest, TTSModel

tts = RapidTTS(model=TTSModel.MOSS_NANO_ONNX)

text = "2026年5月8日，猪肉价格为每斤13.5元，较前期上涨6.3%；车牌号码为京A86F29。"
result = tts.synthesize(SynthesisRequest(text=text))

save_path = "outputs/moss_nano_result.wav"
result.save(save_path)
print(save_path)
