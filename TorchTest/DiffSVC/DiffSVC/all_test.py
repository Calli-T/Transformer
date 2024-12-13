from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from utils.path_utils import rel2abs

'''# 경로 관련 코드
# sys.path.append('.')를 쓰자
hparams["emb_model_path"] = rel2abs(hparams["emb_model_path"])
hparams["wavenet_model_path"] = rel2abs(hparams["wavenet_model_path"])
print(hparams["wavenet_model_path"], hparams["emb_model_path"])'''

diff = GuassianDiffusion(hparams)
# print(diff.alphas)

'''from vocoder.NsfHiFiGAN.nsf_hifigan import NsfHifiGAN

vocoder = NsfHifiGAN(hparams)
wav, mel = vocoder.wav2spec(rel2abs(hparams['raw_wave_path']), hparams)
print(wav.shape, mel.shape)
'''