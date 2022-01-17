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
# -*- encoding: utf-8 -*-
import onnxruntime as ort


class PWGANVocoder(object):
    def __init__(self, model_path):
        sess_opt = ort.SessionOptions()
        sess_opt.log_severity_level = 4
        sess_opt.enable_cpu_mem_arena = False
        self.sess = ort.InferenceSession(model_path,
                                         sess_options=sess_opt)
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, am_output_data):
        wav = self.sess.run(None, {self.input_name: am_output_data})[0]
        return wav
