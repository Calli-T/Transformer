#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained Parallel WaveGAN Generator."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from audio_mel_dataset import MelDataset, PipelineDataset
from utils import load_model, read_hdf5


def mel2wav(model_path, for_config=None, output_path=None, for_dataset=None):
    """Run decoding process."""
    # load config
    dirname = os.path.dirname(model_path)
    config_path = os.path.join(dirname, "config.yml")

    # load config, 파이프라이닝 할 때는 경로가 아니라 config 그 자체를 전달
    if type(config_path) == str:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config = for_config

    '''if for_config is None:
        config = for_config
    else:
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.Loader)'''

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(model_path, config)
    logging.info(f"Loaded model parameters from {model_path}.")

    model.remove_weight_norm()
    model = model.eval().to(device)
    model.to(device)

    # check model type
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_aux_input = "VQVAE" not in generator_type

    # if use_aux_input:
    ############################
    #       MEL2WAV CASE       #
    ############################
    # setup dataset
    if for_dataset is not None:
        if config["format"] == "hdf5":
            mel_query = "*.h5"
            mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
        elif config["format"] == "npy":
            mel_query = "*-feats.npy"
            mel_load_fn = np.load
        else:
            raise ValueError("Support only hdf5 or npy format.")

        if type(for_dataset) == "str":  # not use_f0_and_excitation:
            dataset = MelDataset(
                for_dataset,
                mel_query=mel_query,
                mel_load_fn=mel_load_fn,
                return_utt_id=True,
            )
        else:
            dataset = PipelineDataset(*for_dataset)
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # start generation
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        wav_list = []
        for idx, items in enumerate(pbar, 1):
            if not False:  # use_f0_and_excitation:
                utt_id, c = items
                f0, excitation = None, None
            else:
                utt_id, c, f0, excitation = items
            batch = dict(normalize_before=False)
            if c is not None:
                c = torch.tensor(c, dtype=torch.float).to(device)
                batch.update(c=c)
            if f0 is not None:
                f0 = torch.tensor(f0, dtype=torch.float).to(device)
                batch.update(f0=f0)
            if excitation is not None:
                excitation = torch.tensor(excitation, dtype=torch.float).to(device)
                batch.update(excitation=excitation)
            start = time.time()
            y = model.inference(**batch).view(-1)
            rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            if output_path is not None:
                # check directory existence
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # save as PCM 16 bit wav file
                sf.write(
                    os.path.join(output_path, f"{utt_id}_gen.wav"),
                    y.cpu().numpy(),
                    config["sampling_rate"],
                    "PCM_16",
                )
            else:
                wav_list.append(y.cpu().numpy())

    # report average RTF
    logging.info(
        f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )

    return wav_list


'''params = [
    "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl",
    "files_for_gen/dump/sample/norm/",
    "files_for_gen/outputs/"
]'''
'''print(mel2wav(model_path="files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl",
              dump_path="files_for_gen/dump/sample/norm/")[0].shape)'''
# mel2wav(model_path=params[0], dump_path=params[1], output_path=params[2])