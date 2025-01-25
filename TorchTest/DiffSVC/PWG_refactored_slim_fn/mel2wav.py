#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Normalize feature files_for_gen and dump them."""
# import wav2mel

import argparse
import logging
import os

import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from audio_mel_dataset import AudioMelDataset, PipelineDataset
from utils import read_hdf5, write_hdf5


def normalize(for_config, for_stats, for_dataset=None, dump_path=None):
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Normalize dumped raw features (See detail in"
            " parallel_wavegan/bin/normalize.py)."
        )
    )
    parser.add_argument(
        "--target-feats",
        type=str,
        default="feats",
        choices=["feats", "local"],
        help="target name to be normalized.",
    )
    args = parser.parse_args()

    # load config, 파이프라이닝 할 때는 경로가 아니라 config 그 자체를 전달
    if type(for_config) == str:
        with open(for_config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        config.update(vars(args))
    else:
        config = for_config

    # # check model architecture
    # generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    #
    # # get dataset
    # if config["format"] == "hdf5":
    #     audio_query, mel_query = "*.h5", "*.h5"
    #     audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
    #     mel_load_fn = lambda x: read_hdf5(x, args.target_feats)  # NOQA
    #     if config.get("use_global_condition", False):  # 일단 pwg 모델은 안쓰더라, global
    #         global_query = "*.h5"
    #         global_load_fn = lambda x: read_hdf5(x, "global")  # NOQA
    # elif config["format"] == "npy":
    #     audio_query, mel_query = "*-wave.npy", f"*-{args.target_feats}.npy"
    #     audio_load_fn = np.load
    #     mel_load_fn = np.load
    #
    #     if config.get("use_global_condition", False):
    #         global_query = "*-global.npy"
    #         global_load_fn = np.load
    # else:
    #     raise ValueError("support only hdf5 or npy format.")

    dataset = PipelineDataset(*for_dataset)

    logging.info(f"The number of files_for_gen = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    if type(for_config) == str:
        if config["format"] == "hdf5":
            scaler.mean_ = read_hdf5(for_stats, "mean")
            scaler.scale_ = read_hdf5(for_stats, "scale")
        elif config["format"] == "npy":
            scaler.mean_ = np.load(for_stats)[0]
            scaler.scale_ = np.load(for_stats)[1]
        else:
            raise ValueError("support only hdf5 or npy format.")
    else:
        scaler.mean_, scaler.scale_ = for_stats
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
