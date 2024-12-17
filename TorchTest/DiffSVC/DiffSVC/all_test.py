from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from utils.path_utils import rel2abs

from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

vocoder = NsfHifiGAN(hparams)
'''wav, mel = vocoder.wav2spec(rel2abs(hparams['raw_wave_path']), hparams)
print(wav.shape, mel.shape)'''

'''# 경로 관련 코드
# sys.path.append('.')를 쓰자
hparams["emb_model_path"] = rel2abs(hparams["emb_model_path"])
hparams["wavenet_model_path"] = rel2abs(hparams["wavenet_model_path"])
print(hparams["wavenet_model_path"], hparams["emb_model_path"])'''

diff = GuassianDiffusion(hparams, vocoder.wav2spec)

# raw -> tensor -> collated_tensor -> embedding
'''cond_tensor = diff.get_tensor_cond(diff.get_raw_cond(rel2abs(hparams['raw_wave_path'])))
collated_tensor = diff.get_collated_cond(cond_tensor)
diff.embedding_model.eval()
embedded = diff.embedding_model(collated_tensor)'''

# embedding
'''
embedding = diff.get_cond(rel2abs(hparams['raw_wave_path']))
print(embedding['decoder_inp'].shape)
print(embedding['f0_denorm'].shape)
'''

# spec_min/max (opencpop)
'''
import numpy as np
np.set_printoptions(precision=20)
print(diff.spec_max)
print(type(diff.spec_max))
'''

# outputs
outputs = diff.infer(rel2abs(hparams['raw_wave_path']))
print(outputs.keys())

# get pe model
'''import torch

pe = PitchExtractor(hparams)
pe_state_dict = torch.load(hparams['pe_ckpt'], map_location='cpu')['state_dict']
new_pe_state_dict = {}
for k, v in pe_state_dict.items():
    new_pe_state_dict[k[6:]] = v
pe.load_state_dict(new_pe_state_dict)
pe.to(hparams['device'])

# use pe
pe.eval()
outputs['f0_pred'] = pe(outputs['mel_out'])['f0_denorm_pred'].detach()()'''

import numpy as np
import torch


def after_infer(prediction):
    mel_pred = prediction["mel_out"].to('cpu').numpy()
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
    # -----
    '''audio = []
    audio_sr = 24000
    length = int(np.ceil(len(data) / audio_sr * hparams['audio_sample_rate']))

    fix_audio = np.zeros(length)
    fix_audio[:] = np.mean(wav_pred)
    fix_audio[:len(wav_pred)] = wav_pred[0 if len(wav_pred) < len(fix_audio) else len(wav_pred) - len(fix_audio):]
    audio.extend(list(fix_audio))'''
    import soundfile as sf
    sf.write('results/output.wav', wav_pred, 44100)


after_infer(outputs)
