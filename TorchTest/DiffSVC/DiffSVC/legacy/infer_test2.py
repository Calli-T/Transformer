from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from utils.path_utils import rel2abs

from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

vocoder = NsfHifiGAN(hparams)

diff = GuassianDiffusion(hparams, vocoder.wav2spec)
# outputs
outputs = diff.infer_batch(rel2abs(hparams['raw_wave_path']))

import numpy as np
import torch


def after_infer(prediction):
    mel_preds = prediction["mel_out"].to('cpu').numpy()

    for mel_pred in mel_preds:
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        f0_pred = prediction.get("f0_denorm")

        print(f0_pred.shape)
        f0_pred = f0_pred.to('cpu').numpy()
        print(mel_pred_mask.shape)

        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]

        torch.cuda.is_available() and torch.cuda.empty_cache()

        wav_pred = vocoder.spec2wav(mel_pred, f0=f0_pred)
        print(wav_pred.shape)

        import soundfile as sf
        extension_str = 'flac'
        if prediction["filename"] is not None:
            result_filename = (f'results/{prediction["filename"]}' +
                               f'_{hparams["project_name"]}_{hparams["model_pt_epoch"]}_steps.{extension_str}')
        else:
            result_filename = f'results/{hparams["project_name"]}_{hparams["model_pt_epoch"]}_steps.{extension_str}'
        sf.write(result_filename, wav_pred, 44100)


after_infer(outputs)
