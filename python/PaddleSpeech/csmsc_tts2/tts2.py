# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from pathlib import Path

import soundfile as sf

from acoustic import SpeedySpeechAcoustic
from frontend.zh_frontend import Frontend
from utils import mkdir, read_txt
from vocoder import PWGANVocoder

print('初始化前处理部分')
phones_dict = 'resources/speedyspeech_nosil_baker_ckpt_0.5/phone_id_map.txt'
tones_dict = 'resources/speedyspeech_nosil_baker_ckpt_0.5/tone_id_map.txt'
frontend = Frontend(phone_vocab_path=phones_dict,
                    tone_vocab_path=tones_dict)
print("frontend done!")

print('初始化提取特征模型')
speedyspeech_dir = Path('resources/models/speedyspeech_csmsc')
pdmodel_path = str(speedyspeech_dir / 'speedyspeech_csmsc.pdmodel')
pdiparam_path = str(speedyspeech_dir / 'speedyspeech_csmsc.pdiparams')

am_predictor = SpeedySpeechAcoustic(pdmodel_path, pdiparam_path)
print('am_predictor done!')

print('初始化合成wav模型')
pwgan_model_path = 'resources/models/pwgan_csmsc/pwgan_csmsc.onnx'
voc_predictor = PWGANVocoder(pwgan_model_path)

save_wav_dir = 'infer_result'
mkdir(save_wav_dir)

print('合成指定句子')
sentences_path = 'sentences.txt'
sentences = read_txt(sentences_path)

for sentence_info in sentences:
    start = time.time()

    uuid, sentence = sentence_info.split(' ')

    input_ids = frontend.get_input_ids(sentence,
                                       merge_sentences=True,
                                       get_tone_ids=True)

    am_output_data = am_predictor(input_ids)

    wav = voc_predictor(am_output_data)

    elapse = time.time() - start

    save_wav_path = f'{save_wav_dir}/{uuid}.wav'
    sf.write(save_wav_path, wav, samplerate=24000)

    print(f'{save_wav_path} done!\tcost: {elapse}s')
