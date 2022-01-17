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
from paddle import inference


class SpeedySpeechAcoustic(object):
    def __init__(self, pdmodel_path, pdiparams_path):
        am_config = inference.Config(pdmodel_path, pdiparams_path)
        am_config.disable_glog_info()
        self.am_predictor = inference.create_predictor(am_config)

        self.am_input_names = self.am_predictor.get_input_names()
        self.am_output_names = self.am_predictor.get_output_names()

    def __call__(self, input_ids):
        phone_ids = input_ids["phone_ids"]
        phones = phone_ids[0].numpy()
        phones_handle = self.am_predictor.get_input_handle(self.am_input_names[0])
        phones_handle.reshape(phones.shape)
        phones_handle.copy_from_cpu(phones)

        tone_ids = input_ids["tone_ids"]
        tones = tone_ids[0].numpy()
        tones_handle = self.am_predictor.get_input_handle(self.am_input_names[1])
        tones_handle.reshape(tones.shape)
        tones_handle.copy_from_cpu(tones)

        self.am_predictor.run()
        am_output_handle = self.am_predictor.get_output_handle(self.am_output_names[0])
        am_output_data = am_output_handle.copy_to_cpu()
        return am_output_data
