# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from frontend import English
from frontend.zh_frontend import Frontend


# input
def get_sentences(args):
    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            if 'lang' in args and args.lang == 'zh':
                sentence = "".join(items[1:])
            elif 'lang' in args and args.lang == 'en':
                sentence = " ".join(items[1:])
            sentences.append((utt_id, sentence))
    return sentences


# frontend
def get_frontend(args):
    if 'lang' in args and args.lang == 'zh':
        frontend = Frontend(phone_vocab_path=args.phones_dict,
                            tone_vocab_path=args.tones_dict)
    elif 'lang' in args and args.lang == 'en':
        frontend = English(phone_vocab_path=args.phones_dict)
    else:
        print("wrong lang!")
    print("frontend done!")
    return frontend
