# Speech-Mamba

This is the implementation of the SLT paper Speech-Mamba: Long-Context Speech Recognition with Selective State Spaces Models. This repo can only be used for research purpose. Please find more details in the paper https://arxiv.org/pdf/2409.18654.

### Set Up

We run experiments based on SpeechBrain toolkit. To install SpeechBrain, run following commands:

```bash
git clone https://github.com/speechbrain/speechbrain.git
cd speechbrain
pip install -r requirements.txt
pip install --editable .
```
Please also install Pytorch and other dependencies:
```
pip install sox librosa soundfile
pip install ctc-segmentation
pip install kaldilm
pip install mamba-ssm
```
### Dataset
You can download LibriSpeech at http://www.openslr.org/12

### Train
```
cd ./speechbrain/recipes/LibriSpeech/ASR/transformer
```

Train the Speech-Mamba model with 100 hours of data:
```
python train.py hparams/mamba.yaml
```

Train the Speech-Mamba model with 960 hours of data:
```
python train.py hparams/mamba960h.yaml
```

### Test
```
python train.py hparams/mamba.yaml --test_only
```

In order to test on long-context data, you can create longer test data from 45 seconds to 60 seconds by:
```
python combine_wav.py
```

### Pretrained Models
Please find the pretrained Speech-Mamba models with 100 hours and 960 hours of data in the below link.
https://drive.google.com/drive/folders/1hxinZXQ933thZnLMFpp1v_SUnrcP6EkV?usp=drive_link

### Citation
Please cite the following paper:
```
@article{gao2024speech,
  title={Speech-Mamba: Long-Context Speech Recognition with Selective State Spaces Models},
  author={Gao, Xiaoxue and Chen, Nancy F},
  journal={arXiv preprint arXiv:2409.18654},
  year={2024}
}
