from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN
import numpy as np
import torch

vocoder = NsfHifiGAN(hparams)


# 정상화
def after_infer(prediction):
    mel_preds = prediction["mel_out"].to('cpu').numpy()
    filenames = prediction["filename"]
    f0_preds = prediction["f0_denorm"].to('cpu').numpy()

    for mel_pred, filename, f0_pred in zip(mel_preds, filenames, f0_preds):
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        print(f0_pred.shape)
        print(mel_pred_mask.shape)

        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]

        torch.cuda.is_available() and torch.cuda.empty_cache()

        wav_pred = vocoder.spec2wav(mel_pred, f0=f0_pred)
        print(wav_pred.shape)

        import soundfile as sf
        extension_str = 'flac'
        if filename is not None:
            result_filename = (f'results/{filename}' +
                               f'_{hparams["project_name"]}_{hparams["model_pt_epoch"]}_steps.{extension_str}')
        else:
            result_filename = f'results/{hparams["project_name"]}_{hparams["model_pt_epoch"]}_steps.{extension_str}'
        sf.write(result_filename, wav_pred, 44100)


wav_fname_list = ["raw/L-O-V-E-[cut_12sec].wav", "raw/L-O-V-E_[cut_6sec].wav"]

diff = GuassianDiffusion(hparams, NsfHifiGAN.wav2spec)

outputs = diff.infer_batch(wav_fname_list)
print(outputs.keys())
# print(ret['mel_out'].shape)

after_infer(outputs)
