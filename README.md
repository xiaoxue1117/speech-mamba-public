# Speech-Mamba

This is the implementation of the SLT paper Speech-Mamba: Long-Context Speech Recognition with Selective State Spaces Models.

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
cd ./speechbrain/recipes/LibriSpeech/ASR/transformer
python train.py hparams/mamba.yaml

### Test
python train.py hparams/mamba_test.yaml --test_only

### Citation
Please cite the following paper:
```
@article{gao2024speech,
  title={Speech-Mamba: Long-Context Speech Recognition with Selective State Spaces Models},
  author={Gao, Xiaoxue and Chen, Nancy F},
  journal={arXiv preprint arXiv:2409.18654},
  year={2024}
}
