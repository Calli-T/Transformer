'''
import parallel_wavegan.utils

for unit in dir(parallel_wavegan.utils):
    print(unit)
'''
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("vctk_parallel_wavegan.v1", "pretrained_model")