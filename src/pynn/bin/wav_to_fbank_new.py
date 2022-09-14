#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Thai-Son Nguyen
# Licensed under the Apache License, Version 2.0 (the "License")

import argparse
import glob
import multiprocessing
import os
import random

import numpy as np
import torch
import torchaudio
import tqdm
from pynn.io import kaldi_io
from pynn.util import audio
from torchaudio import functional as F
from torchaudio.sox_effects import apply_effects_tensor


def load(wav, sample_rate):
    wav = wav if wav.endswith(args.suffix) else wav + args.suffix
    signal, sr = torchaudio.load(wav, normalize=True)
    if signal.shape[0] > 1:
        signal = signal.mean(0, keepdim=True)
    if sr != sample_rate:
        signal = F.resample(signal,
                            sr,
                            sample_rate,
                            lowpass_filter_width=64,
                            rolloff=0.9475937167399596,
                            resampling_method="kaiser_window",
                            beta=14.769656459379492
                            )
    return signal


def rir(signal, args):
    rir_wav = args.rirs[random.randint(0, len(args.rirs) - 1)]
    rir = load(rir_wav, args.sample_rate)
    rir /= torch.norm(rir, p=2)
    rir = torch.flip(rir, [1])
    signal = torch.nn.functional.pad(signal, (rir.shape[1] - 1, 0))
    return torch.nn.functional.conv1d(signal[None, ...], rir[None, ...])[0]


def background(signal, args):
    snr_db = args.min_background_snr_db + \
        random.random() * (args.max_background_snr_db - args.min_background_snr_db)
    bck = args.backgrounds[random.randint(0, len(args.backgrounds) - 1)]
    noise = load(bck, args.sample_rate)
    speech_rms = signal.norm(p=2)
    noise_rms = noise.norm(p=2)
    snr = 10 ** (snr_db / 20)
    scale = snr * noise_rms / speech_rms
    l = min(noise.shape[1], signal.shape[1])
    so, no = 0, 0
    if noise.shape[1] > signal.shape[1]:
        no = random.randint(0, noise.shape[1] - signal.shape[1])
    else:
        so = random.randint(0, - noise.shape[1] + signal.shape[1])
    signal[:, so:so+l] = (scale * signal[:, so:so+l] +
                          noise[:, no:no+l]) / (1+scale)
    return signal


def write_ark_thread(segs, out_ark, out_scp, args, progress):
    import torch
    torch.set_num_threads(1)
    fbank_mat = audio.filter_bank(args.sample_rate, args.nfft, args.fbank)
    cache_wav = ''

    ark_file = open(out_ark, 'wb')
    scp_file = open(out_scp, 'w')
    for seg in tqdm.tqdm(segs) if progress else segs:
        tokens = seg.split()

        if len(tokens) == 1:
            tokens.insert(0, '')
        if len(tokens) == 2:
            tokens.extend(['0.0', '0.0'])
        seg_name, wav, start, end = tokens[:4]

        start, end = float(start), float(end)

        if args.wav_path is not None:
            wav = args.wav_path + '/' + wav

        if seg_name == '':
            seg_name = os.path.basename(wav)[:-4]

        if cache_wav != wav:
            if not os.path.isfile(wav):
                print('File %s does not exist' % wav)
                continue
            signal = load(wav, args.sample_rate)
            cache_wav = wav

        def write(signal, start, end, sample_rate, seg_name, fbank_mat, args, suffix=''):
            end = float(signal.shape[1]) / sample_rate if end <= 0. else end
            if args.seg_info:
                seg_name = '%s-%06.f-%06.f' % (seg_name, start*100, end*100)
            start, end = int(start * sample_rate), int(end * sample_rate)
            if start >= signal.shape[1] or start >= end >= 0:
                print('Wrong segment %s' % seg_name)
                return False

            feats = torchaudio.compliance.kaldi.fbank(
                signal,
                num_mel_bins=args.fbank,
                sample_frequency=args.sample_rate,
            ).numpy()
            if len(feats) > args.max_len or len(feats) < args.min_len:
                return False
            if args.mean_norm:
                feats = feats - feats.mean(axis=0, keepdims=True)
            if args.fp16:
                feats = feats.astype(np.float16)

            dic = {seg_name + suffix: feats}
            #kaldi_io.write_ark(out_ark, dic, out_scp, append=True)
            kaldi_io.write_ark_file(ark_file, scp_file, dic)

        if args.apply_effects:
            if args.background_prob > 0:
                signal = background(signal, args)
            if args.rir_prob > 0:
                signal = rir(signal, args)
            signal = F.apply_codec(
                signal, args.sample_rate, compression=random.randint(-1, args.max_ogg_compression), format='ogg', channels_first=True)
            if not write(signal, start, end, args.sample_rate, seg_name, fbank_mat, args, '-effets'):
                continue
        else:
            if not write(signal, start, end, args.sample_rate, seg_name, fbank_mat, args):
                continue

    ark_file.close()
    scp_file.close()


parser = argparse.ArgumentParser(description='pynn')
parser.add_argument(
    '--seg-desc', help='input segment description file', required=True)
parser.add_argument(
    '--seg-info', help='append timestamp suffix to segment name', action='store_true')
parser.add_argument('--wav-path', help='path to wav files',
                    type=str, default=None)
parser.add_argument('--sample-rate', help='sample rate',
                    type=int, default=16000)
parser.add_argument(
    '--fbank', help='number of filter banks', type=int, default=40)
parser.add_argument('--nfft', help='number of FFT points',
                    type=int, default=256)
parser.add_argument(
    '--max-len', help='maximum frames for a segment', type=int, default=10000)
parser.add_argument(
    '--min-len', help='minimum frames for a segment', type=int, default=4)
parser.add_argument('--mean-norm', help='mean substraction',
                    action='store_true')
parser.add_argument(
    '--fp16', help='use float16 instead of float32', action='store_true')
parser.add_argument('--output', help='output file', type=str, default='data')
parser.add_argument(
    '--jobs', help='number of parallel jobs', type=int, default=1)

parser.add_argument('--suffix', help='wav suffix', type=str, default='.wav')

parser.add_argument('--apply-effects', action='store_true')

parser.add_argument(
    '--rirs', help='wildcards to RIR recordings', type=str, default=None)
parser.add_argument('--rir-prob', help='RIR prob', type=float, default=0.)


parser.add_argument(
    '--backgrounds', help='wildcards to background recordings', type=str, default=None)
parser.add_argument('--background-prob',
                    help='background prob', type=float, default=0.)
parser.add_argument('--max-background-snr-db',
                    help='max background snr-db ration', type=float, default=20)

parser.add_argument('--min-background-snr-db',
                    help='min background snr-db ration', type=float, default=5)

parser.add_argument('--max-ogg-compression',
                    help='background prob', type=float, default=5)


if __name__ == '__main__':
    args = parser.parse_args()

    segs = [line.rstrip('\n') for line in open(args.seg_desc, 'r')]

    if args.rirs:
        args.rirs = list(glob.glob(args.rirs))
        print(f'Found {len(args.rirs)} RIRs')

    if args.backgrounds:
        args.backgrounds = list(glob.glob(args.backgrounds))
        print(f'Found {len(args.backgrounds)} backgrounds')
    elif args.background_prob > 0:
        args.backgrounds = [os.path.join(
            args.wav_path, s.split(maxsplit=2)[-1]) for s in segs]

    size = len(segs) // args.jobs
    jobs = []
    j = 0
    for i in range(args.jobs):
        l = len(segs) if i == (args.jobs-1) else j+size
        sub_segs = segs[j:l]
        j += size
        out_ark = '%s.%d.ark' % (args.output, i)
        out_scp = '%s.%d.scp' % (args.output, i)

        process = multiprocessing.Process(
            target=write_ark_thread, args=(sub_segs, out_ark, out_scp, args, i == 0))
        process.start()
        jobs.append(process)

    for job in jobs:
        job.join()
