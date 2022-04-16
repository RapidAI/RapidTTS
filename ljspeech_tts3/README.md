### ljspeech_tts3
- **支持合成语言**: 英文字母
- 基于[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)下的[ljspeech-TTS3](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/ljspeech/tts3/README.md)整理而来
- 整个推理引擎只采用`ONNXRuntime`
- 其中PaddleSpeech中提供的预训练模型可以参见[link](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/demos/text_to_speech/README_cn.md#4-%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B)。在ljspeech_tts3中使用的是:

    |主要部分|具体模型|支持语言|
    |:---|:---|:---|
    |声学模型|[fastspeech2_ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/examples/ljspeech/tts3/README.md#pretrained-model)|en|
    |声码器|[pwg_ljspeech](https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/examples/ljspeech/voc1)|en|

#### 结果示例
<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th width="550">输入文本</th>
      <th>合成音频</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td >Love you three thousand times.</td>
      <td align = "center">
      <a href="https://drive.google.com/file/d/1RkpE569HBou7__568HupQlo3w21pAysR/view?usp=sharing" rel="nofollow" target="_blank">
            <img align="center" src="./assets/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</table>

</div>

#### 运行步骤
1. 下载`resources`, [Google Drive](https://drive.google.com/file/d/1xQwsY1tWebQSWu32KgLlGO1QUnrixvwo/view?usp=sharing) | [百度网盘,提取码:4vlu](https://pan.baidu.com/s/1vvBnuNEcj-AngXdw3j0S4g?pwd=4vlu), 解压到`ljspeech_tts3`目录下，最终目录结构如下：
   ```text
    ljspeech_tts3
    ├── sentences_en.txt
    ├── requirements.txt
    ├── frontend
    ├── main.sh
    ├── tts3.py
    ├── infer_result
    ├── resources
    │   ├── fastspeech2_ljspeech
    │   │   ├── fastspeech2_ljspeech.onnx
    │   │   └── phone_id_map.txt
    │   └── pwgan_ljspeech.onnx
    └──syn_utils.py
   ```

2. 安装`requirements.txt`
   ```bash
   pip install -r requirements.txt -i https://pypi.douban.com/simple/
   ```

3. 运行`tts3.py`
   ```bash
   python tts3.py
   ```
   or
   ```bash
   bash main.sh
   ```

4. 运行日志如下:
   ```text
    frontend done!
    001, mel: (343, 80), wave: 87808, time: 7.583922399999999s, Hz: 11578.186242472837, RTF: 1.9044433677455357.
    002, mel: (274, 80), wave: 70144, time: 5.986744399999999s, Hz: 11716.561243394675, RTF: 1.8819514994154878.
    003, mel: (175, 80), wave: 44800, time: 3.911470399999999s, Hz: 11453.51349948683, RTF: 1.9251734414062498.
    004, mel: (217, 80), wave: 55552, time: 4.678628299999996s, Hz: 11873.585640758554, RTF: 1.8570632888104823.
    005, mel: (371, 80), wave: 94976, time: 7.7152417s, Hz: 12310.185834993608, RTF: 1.7911996045843162.
    006, mel: (338, 80), wave: 86528, time: 7.670878100000003s, Hz: 11280.071739420744, RTF: 1.954774801913832.
    007, mel: (205, 80), wave: 52480, time: 4.628822800000002s, Hz: 11337.668363997142, RTF: 1.9448443270769813.
    008, mel: (390, 80), wave: 99840, time: 8.2700763s, Hz: 12072.447745611855, RTF: 1.826473012319712.
    009, mel: (169, 80), wave: 43264, time: 4.2657806000000065s, Hz: 10142.12548840801, RTF: 2.1741004905926427.
    generation speed: 11613.502408804885Hz, RTF: 1.8986520365538124
   ```
   生成结果会保存到`infer_result`目录下
