import numpy as np
import torch


def after_infer(prediction, _vocoder, _hparams):
    mel_preds = prediction["mel_out"].to('cpu').numpy()
    filenames = prediction["filename"]
    f0_preds = prediction["f0_denorm"].to('cpu').numpy()

    for mel_pred, filename, f0_pred in zip(mel_preds, filenames, f0_preds):
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, _hparams['mel_vmin'], _hparams['mel_vmax'])

        print(f0_pred.shape)
        print(mel_pred_mask.shape)

        if len(f0_pred) > len(mel_pred_mask):
            f0_pred = f0_pred[:len(mel_pred_mask)]
        f0_pred = f0_pred[mel_pred_mask]

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
