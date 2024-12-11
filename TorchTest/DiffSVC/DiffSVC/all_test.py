from args2hparams import hparams
from diffusion.diffusion import GuassianDiffusion
from utils.path_utils import rel2abs

# 절대경로 관련 코드
hparams["emb_model_path"] = rel2abs(hparams["emb_model_path"])
hparams["wavenet_model_path"] = rel2abs(hparams["wavenet_model_path"])
print(hparams["wavenet_model_path"], hparams["emb_model_path"])

diff = GuassianDiffusion(hparams)
# print(diff.alphas)