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

'''
embedding = diff.get_cond(rel2abs(hparams['raw_wave_path']))
print(embedding['decoder_inp'].shape)
print(embedding['f0_denorm'].shape)
'''

sample = diff.infer(rel2abs(hparams['raw_wave_path']))
print(sample.shape)
