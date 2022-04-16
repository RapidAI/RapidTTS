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
    warm up done!
    Building prefix dict from the default dictionary ...
    Loading model from cache C:\Users\WANGJI~1\AppData\Local\Temp\jieba.cache
    Loading model cost 0.836 seconds.
    Prefix dict has been built successfully.
    009901, mel: (331, 80), wave: 99300, time: 1.3718173s, Hz: 72385.938204132, RTF: 0.33155610876132857.
    009902, mel: (288, 80), wave: 86400, time: 1.1350326000000024s, Hz: 76121.49025085453, RTF: 0.3152854722222228.
    009903, mel: (341, 80), wave: 102300, time: 1.4687841000000006s, Hz: 69649.7502651354, RTF: 0.3445812785923755.
    generation speed: 72441.68237053939Hz, RTF: 0.33130097499999983
   ```
   生成结果会保存到`infer_result`目录下
