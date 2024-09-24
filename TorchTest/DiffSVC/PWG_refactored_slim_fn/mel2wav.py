#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalize feature files_for_gen and dump them."""

import argparse
import logging
import os

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from audio_mel_dataset import (
    AudioMelDataset,
)
from utils import read_hdf5, write_hdf5

'''
--config files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml \
    --rootdir files_for_gen/dump/sample/raw \
    --dumpdir files_for_gen/dump/sample/norm \
    --stats files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5
'''


def normalize(config_path, raw_path, stats_path, dump_path=None):
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize dumped raw features (See detail in"
            " parallel_wavegan/bin/normalize.py)."
        )
    )

    '''parser.add_argument(
        "--skip-wav-copy",
        default=False,
        action="store_true",
        help="whether to skip the copy of wav files_for_gen.",
    )'''
    parser.add_argument(
        "--target-feats",
        type=str,
        default="feats",
        choices=["feats", "local"],
        help="target name to be normalized.",
    )

    args = parser.parse_args()

    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    # )

    # load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # check model architecture
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")

    # get dataset
    if raw_path is not None:
        global_query = None
        global_load_fn = None
        if config["format"] == "hdf5":
            audio_query, mel_query = "*.h5", "*.h5"
            audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
            mel_load_fn = lambda x: read_hdf5(x, args.target_feats)  # NOQA
            if config.get("use_global_condition", False):
                global_query = "*.h5"
                global_load_fn = lambda x: read_hdf5(x, "global")  # NOQA
        elif config["format"] == "npy":
            audio_query, mel_query = "*-wave.npy", f"*-{args.target_feats}.npy"
            audio_load_fn = np.load
            mel_load_fn = np.load

            if config.get("use_global_condition", False):
                global_query = "*-global.npy"
                global_load_fn = np.load
        else:
            raise ValueError("support only hdf5 or npy format.")

        dataset = AudioMelDataset(
            root_dir=raw_path,
            audio_query=audio_query,
            mel_query=mel_query,
            audio_load_fn=audio_load_fn,
            mel_load_fn=mel_load_fn,
            global_query=global_query,
            global_load_fn=global_load_fn,
            return_utt_id=True,
        )

    logging.info(f"The number of files_for_gen = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    if config["format"] == "hdf5":
        scaler.mean_ = read_hdf5(stats_path, "mean")
        scaler.scale_ = read_hdf5(stats_path, "scale")
    elif config["format"] == "npy":
        scaler.mean_ = np.load(stats_path)[0]
        scaler.scale_ = np.load(stats_path)[1]
    else:
        raise ValueError("support only hdf5 or npy format.")
    # from version 0.23.0, this information is needed
    scaler.n_features_in_ = scaler.mean_.shape[0]

    # process each file
    mel_norm_list = []
    for items in tqdm(dataset):
        if config.get("use_global_condition", False):
            utt_id, audio, mel, g = items
        else:

            utt_id, audio, mel = items

        # normalize
        mel_norm = scaler.transform(mel)

        # replace with the original features if the feature is binary
        if args.target_feats == "local":
            is_binary = np.logical_or(mel == 1, mel == 0).sum(axis=0) == len(mel)
            for idx, isb in enumerate(is_binary):
                if isb:
                    mel_norm[:, idx] = mel[:, idx]

        if dump_path is not None:
            # check directory existence
            if not os.path.exists(dump_path):
                os.makedirs(dump_path)

            # save
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(dump_path, f"{utt_id}.h5"),
                    args.target_feats,
                    mel_norm.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(dump_path, f"{utt_id}.h5"),
                    "wave",
                    audio.astype(np.float32),
                )
                if config.get("use_global_condition", False):
                    write_hdf5(
                        os.path.join(dump_path, f"{utt_id}.h5"), "global", g.reshape(-1)
                    )
            elif config["format"] == "npy":
                np.save(
                    os.path.join(dump_path, f"{utt_id}-{args.target_feats}.npy"),
                    mel_norm.astype(np.float32),
                    allow_pickle=False,
                )
                if True:  # not args.skip_wav_copy:
                    np.save(
                        os.path.join(dump_path, f"{utt_id}-wave.npy"),
                        audio.astype(np.float32),
                        allow_pickle=False,
                    )
                if config.get("use_global_condition", False):
                    np.save(
                        os.path.join(dump_path, f"{utt_id}-global.npy"),
                        g.reshape(-1),
                        allow_pickle=False,
                    )
            else:
                raise ValueError("support only hdf5 or npy format.")
        else:
            mel_norm_list.append(mel_norm)

    return mel_norm_list


'''print(len(normalize(raw_path="files_for_gen/dump/sample/raw/",
          stats_path="files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5",
          config_path="files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml")))'''

normalize(raw_path="files_for_gen/dump/sample/raw/", dump_path="files_for_gen/dump/sample/norm/",
          stats_path="files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/stats.h5",
          config_path="files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml")
