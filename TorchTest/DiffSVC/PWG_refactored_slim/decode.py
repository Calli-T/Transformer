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

from utils.audio_mel_dataset import MelDataset
from utils.utils import load_model, read_hdf5


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode dumped features with trained Parallel WaveGAN Generator "
            "(See detail in parallel_wavegan/bin/decode.py)."
        )
    )

    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files_for_gen. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--normalize-before",
        default=False,
        action="store_true",
        help=(
            "whether to perform feature normalization before input to the model. if"
            " true, it assumes that the feature is de-normalized. this is useful when"
            " text2mel model and vocoder use different feature statistics."
        ),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # setup model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = load_model(args.checkpoint, config)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")
    if args.normalize_before:
        assert hasattr(model, "mean"), "Feature stats are not registered."
        assert hasattr(model, "scale"), "Feature stats are not registered."
    model.remove_weight_norm()
    model = model.eval().to(device)
    model.to(device)

    # check model type
    generator_type = config.get("generator_type", "ParallelWaveGANGenerator")
    use_aux_input = "VQVAE" not in generator_type

    if use_aux_input:
        ############################
        #       MEL2WAV CASE       #
        ############################
        # setup dataset
        if args.dumpdir is not None:
            if config["format"] == "hdf5":
                mel_query = "*.h5"
                mel_load_fn = lambda x: read_hdf5(x, "feats")  # NOQA
            elif config["format"] == "npy":
                mel_query = "*-feats.npy"
                mel_load_fn = np.load
            else:
                raise ValueError("Support only hdf5 or npy format.")

            if True:  # not use_f0_and_excitation:
                dataset = MelDataset(
                    args.dumpdir,
                    mel_query=mel_query,
                    mel_load_fn=mel_load_fn,
                    return_utt_id=True,
                )
        logging.info(f"The number of features to be decoded = {len(dataset)}.")

        # start generation
        total_rtf = 0.0
        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            for idx, items in enumerate(pbar, 1):
                if not False:  # use_f0_and_excitation:
                    utt_id, c = items
                    f0, excitation = None, None
                else:
                    utt_id, c, f0, excitation = items
                batch = dict(normalize_before=args.normalize_before)
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

                # save as PCM 16 bit wav file
                sf.write(
                    os.path.join(config["outdir"], f"{utt_id}_gen.wav"),
                    y.cpu().numpy(),
                    config["sampling_rate"],
                    "PCM_16",
                )

        # report average RTF
        logging.info(
            f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f})."
        )


if __name__ == "__main__":
    main()
