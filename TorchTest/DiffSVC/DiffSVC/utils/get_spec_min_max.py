import numpy as np
import os


# hparams의 training data set의 raw에 wav를 가져다가 spec_min/max배열을 준다
def get_spec_min_max(_hparams, _wav2spec):
    fname = _hparams['train_dataset_path_spec_minmax']
    fname = os.path.join(fname, 'minmax.npz')

    if os.path.isfile(fname):
        temp = np.load(fname)
        print('mel-spectrogram 값 범위 파일 읽는 중')
        spec_min = temp['spec_min']
        spec_max = temp['spec_max']
    else:
        dir_path = _hparams['train_dataset_path_input']
        raw_wav_path_list = os.listdir(dir_path)

        min_list = []
        max_list = []
        for raw_wav_path in raw_wav_path_list:
            temp_path = os.path.join(dir_path, raw_wav_path)
            wav, mel = _wav2spec(temp_path, _hparams)
            spec_min = np.min(mel, axis=0)
            spec_max = np.max(mel, axis=0)
            min_list.append(spec_min)
            max_list.append(spec_max)

        min_list = np.array(min_list)
        max_list = np.array(max_list)
        spec_min = np.min(min_list, axis=0)
        spec_max = np.max(max_list, axis=0)

        np.savez(fname, spec_min=spec_min, spec_max=spec_max)
        print('mel-spectrogram 값 범위 파일 생성됨')

    '''for spec_c in spec_max:
        print(f'- {spec_c:.20f}')'''

    # spec_min_str = '['
    # for spec_c in spec_min:
    #     spec_min_str = spec_min_str + str(f'{spec_c:.20}') + ', '
    # spec_min_str = spec_min_str + ']'
    # spec_max_str = '['
    # for spec_c in spec_max:
    #     spec_max_str = spec_max_str + str(f'{spec_c:.20}') + ', '
    # spec_max_str = spec_max_str + ']'
    #
    # # print(spec_min_str)
    # print(spec_max_str)
    # # print(spec_min.shape)
    return spec_min, spec_max
