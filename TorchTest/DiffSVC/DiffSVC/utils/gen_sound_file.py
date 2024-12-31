import numpy as np
import torch


def after_infer(prediction, _vocoder, _hparams):
    mel_preds = prediction["mel_out"].to('cpu').numpy()
    filenames = prediction["filename"]
    f0_preds = prediction["f0_denorm"].to('cpu').numpy()
    mels_len = prediction["mel_len"]

    for mel_pred, filename, f0_pred, mel_len in zip(mel_preds, filenames, f0_preds, mels_len):
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = mel_pred[:mel_len, :]
        mel_pred = np.clip(mel_pred, _hparams['mel_vmin'], _hparams['mel_vmax'])

        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]
        f0_pred = f0_pred[:mel_len]

        torch.cuda.is_available() and torch.cuda.empty_cache()

        wav_pred = _vocoder.spec2wav(mel_pred, f0=f0_pred)
        print(wav_pred.shape)

        import soundfile as sf
        extension_str = 'flac'
        if filename is not None:
            result_filename = (f'results/{filename}' +
                               f'_{_hparams["project_name"]}_{_hparams["model_pt_epoch"]}_steps.{extension_str}')
        else:
            result_filename = f'results/{_hparams["project_name"]}_{_hparams["model_pt_epoch"]}_steps.{extension_str}'
        sf.write(result_filename, wav_pred, 44100)
