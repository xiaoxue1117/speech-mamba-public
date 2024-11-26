import torch
import torchaudio
from collections import defaultdict
import numpy
import os

min_whole_audio_length = 45
max_whole_audio_length = 60
wav_info = './data/librispeech_wav_sort/dev-other_sort.csv'
out_dir = './datasets/long_librispeech/dev-other45_60/'
out_meta_file = '/speechbrain/recipes/LibriSpeech/ASR/transformer/data/librispeech_long_train_bk/dev-other45_60.csv'
try:
    os.mkdir(out_dir)
except:
    print("folder exists")

def combine_wav(wav_list):
    combined_wav = torch.cat(wav_list, dim=1)
    return combined_wav

with open(wav_info,'r') as f:
    wav_info_lines = f.readlines()

speaker_dict = defaultdict(list)
for line in wav_info_lines:
    id, dur, wavid, spk, text = line.split(',')
    speaker_dict[spk].append((id, float(dur), wavid, spk, text))
grouped_data = dict(speaker_dict)

save_meta_all = ''
drop_sentence = 0
for speaker, content in grouped_data.items():
    total_dur_per_spk = numpy.sum([(l[1]) for l in content])
    num_combine_audio_per_spk = int(total_dur_per_spk/max_whole_audio_length)
    init_duration = 0
    concat_wav_list = []
    concat_text = ''
    for idx, lines in enumerate(content):
        cur_dur = lines[1]
        cur_text = lines[4].strip()
        cur_spk = lines[3]
        init_duration = init_duration + cur_dur
        if max_whole_audio_length >= init_duration >= min_whole_audio_length:
            #continue concat and save combined audio"
            each_line_wav, sr = torchaudio.load(lines[2])
            concat_wav_list.append(each_line_wav)
            combined_wav_cur = combine_wav(concat_wav_list)
            sub_dir1 = lines[0].split('-')[0]
            sub_dir2 = lines[0].split('-')[1]
            save_wav_name = out_dir + sub_dir1 + os.sep + sub_dir2 + os.sep + lines[0] + 'combine.flac'
            concat_text = concat_text + ' ' + cur_text
            save_meta = lines[0] + 'combine' + ',' + str(init_duration) + ',' + save_wav_name + ',' + cur_spk + ',' + concat_text + '\n'
            save_meta_all = save_meta_all + save_meta
            try:
                os.mkdir(out_dir + sub_dir1)
            except:
                print("folder exists")
            try:
                os.mkdir(out_dir + sub_dir1 + os.sep + sub_dir2)
            except:
                print("folder exists")
            torchaudio.save(save_wav_name, combined_wav_cur, 16000)
            init_duration = 0
            concat_wav_list = []
            concat_text = ''
        elif init_duration < min_whole_audio_length:
            # continue concat
            each_line_wav, sr = torchaudio.load(lines[2])
            concat_wav_list.append(each_line_wav)
            concat_text = concat_text + ' ' + cur_text
        else:
            # audio length > max length
            if idx + 1 == len(content): #if last one, drop it
                init_duration = 0
                concat_wav_list = []
                concat_text = ''
                drop_sentence = drop_sentence + 1
            else: #leave the current audio for next round
                init_duration = cur_dur
                concat_wav_list = []
                concat_text = cur_text
                each_line_wav, sr = torchaudio.load(lines[2])
                concat_wav_list.append(each_line_wav)


with open(out_meta_file, 'w') as f:
    f.write(save_meta_all)







