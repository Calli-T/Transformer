import numpy as np


def get_align(mel, phone_encoded):
    mel2ph = np.zeros([mel.shape[0]], int)
    start_frame = 0
    ph_durs = mel.shape[0] / phone_encoded.shape[0]
    # if hparams['debug']:
    #     print(mel.shape, phone_encoded.shape, mel.shape[0] / phone_encoded.shape[0])
    for i_ph in range(phone_encoded.shape[0]):
        end_frame = int(i_ph * ph_durs + ph_durs + 0.5)
        mel2ph[start_frame:end_frame + 1] = i_ph + 1
        start_frame = end_frame + 1

    return mel2ph
