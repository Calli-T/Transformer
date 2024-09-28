#!/usr/bin/env python3

# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import logging
import os

import librosa
import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm

from audio_mel_dataset import AudioDataset
from utils import write_hdf5


def logmelfilterbank(
        audio,
        sampling_rate,
        fft_size=1024,
        hop_size=256,
        win_length=None,
        window="hann",
        num_mels=80,
        fmin=None,
        fmax=None,
        eps=1e-10,
        log_base=10.0,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )

    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


# 만약 dump_path가 None이라면, 저장을 안하고 넘파이 배열만 돌려주는 것으로 하자
def wav2mel(sample_path, for_config, dump_path=None):
    # load config, 파이프라이닝 할 때는 경로가 아니라 config 그 자체를 전달
    if type(for_config) == str:
        with open(for_config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
    else:
        config = for_config

    # check model architecture
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")

    # get dataset
    if sample_path is not None:
        dataset = AudioDataset(
            sample_path,
            "*.wav",
            audio_load_fn=sf.read,
            return_utt_id=True,
        )

    if "sampling_rate_for_feats" not in config:
        sampling_rate = config["sampling_rate"]
    else:
        sampling_rate = config["sampling_rate_for_feats"]

    # process each data
    mel_list = []
    utt_id_list = []
    audio_list = []
    for utt_id, (audio, fs) in tqdm(dataset):
        # check
        assert len(audio.shape) == 1, f"{utt_id} seems to be multi-channel signal."
        assert (
                np.abs(audio).max() <= 1.0
        ), f"{utt_id} seems to be different from 16 bit PCM."
        assert (
                fs == config["sampling_rate"]
        ), f"{utt_id} seems to have a different sampling rate."

        # trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        if "sampling_rate_for_feats" not in config:
            x = audio
            sampling_rate = config["sampling_rate"]
            hop_size = config["hop_size"]

        # extract feature
        mel = logmelfilterbank(
            x,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=config["window"],
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
        )

        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
        audio = audio[: len(mel) * config["hop_size"]]
        assert len(mel) * config["hop_size"] == len(audio)

        # apply global gain
        if config["global_gain_scale"] > 0.0:
            audio *= config["global_gain_scale"]
        if np.abs(audio).max() >= 1.0:
            logging.warn(
                f"{utt_id} causes clipping. "
                "it is better to re-consider global gain scale."
            )
            continue

        if dump_path is not None:
            # check directly existence
            if not os.path.exists(dump_path):
                os.makedirs(dump_path, exist_ok=True)

            # save
            if config["format"] == "hdf5":
                write_hdf5(
                    os.path.join(dump_path, f"{utt_id}.h5"),
                    "wave",
                    audio.astype(np.float32),
                )
                write_hdf5(
                    os.path.join(dump_path, f"{utt_id}.h5"),
                    "feats",
                    mel.astype(np.float32),
                )

            elif config["format"] == "npy":
                np.save(
                    os.path.join(dump_path, f"{utt_id}-wave.npy"),
                    audio.astype(np.float32),
                    allow_pickle=False,
                )
                if True:  # args.skip_mel_ext:
                    np.save(
                        os.path.join(dump_path, f"{utt_id}-feats.npy"),
                        mel.astype(np.float32),
                        allow_pickle=False,
                    )

            else:
                raise ValueError("support only hdf5 or npy format.")
        else:
            utt_id_list.append(utt_id)
            mel_list.append(mel)
            audio_list.append(audio)

    if dump_path is None:
        return [utt_id_list, audio_list,
                mel_list]  # , PipelineDataset(utt_ids=utt_id_list, waves=audio_list, mels=mel_list)


'''params = [
    "files_for_gen/sample/",
    "files_for_gen/pretrained_model/vctk_parallel_wavegan.v1.long/config.yml",
    "files_for_gen/dump/sample/raw",
]'''

# dump_path=None, 넘파이 배열를 return, 경로가 있을 경우 거기다 저장
# print(len(wav2mel(sample_path=sample_path, config_path=config_path)))

'''for item in wav2mel(sample_path=params[0], config_path=params[1]):
    print(item[1].shape)
    print(item[2].shape)'''

# name, ad, mel = wav2mel(sample_path=params[0], for_config=params[1])
# print(mel[0].shape)

# wav2mel(sample_path=sample_path, config_path=config_path, dump_path=dump_path)
