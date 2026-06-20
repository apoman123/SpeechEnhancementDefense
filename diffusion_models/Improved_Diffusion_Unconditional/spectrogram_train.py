"""
Train a diffusion model on images.
"""

import argparse, os

from improved_diffusion import dist_util, logger
from improved_diffusion.sc09_spectrogram_dataset import load_sc09_data
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop

from datasets import load_from_disk, Audio
from torch.utils.data import DataLoader
import torch

def amp_to_db(x):
    return 20.0 * torch.log10(torch.max(torch.tensor(1e-5), x))

def normalize(S):
    return torch.clip(S / 100, -1.0, 0.0) + 1.0

def wav2spec(wav):
    if len(wav.shape) == 3:
        wav = wav.squeeze(1)
    D = torch.stft(wav, n_fft=128, return_complex=True).unsqueeze(1)
    S = amp_to_db(torch.abs(D)) - 20
    S, D = normalize(S), torch.angle(D)
    return S, D

def collate(samples):
    input_values = [torch.from_numpy(sample['audio']['array']).float() for sample in samples]
    lens = [value.shape[-1] for value in input_values]
    max_len = max(lens)
    input_values = [
        torch.cat([
            value,
            torch.zeros(max_len - value.shape[-1])
        ], dim=-1) for value in input_values
    ]

    labels = torch.tensor([sample['label'] for sample in samples])
    input_values = torch.stack(input_values, dim=0)
    spec, phase = wav2spec(input_values)
    # input_values = torchaudio.functional.vad(input_values, 16000)[:, :64000] if torchaudio.functional.vad(input_values, 16000).shape[-1] != 0 else input_values
    return spec, {}
    

def main():
    args = create_argparser().parse_args()

    os.environ["OPENAI_LOGDIR"] = args.save_dir
    print('checkpoints and models will be saved at: {}'.format(os.environ["OPENAI_LOGDIR"]))
    
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    # data = load_sc09_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     n_mels=args.image_size,
    #     class_cond=args.class_cond,
    # )

    dataset = load_from_disk(args.data_dir).cast_column("audio", Audio(sampling_rate=16000))['train']
    data = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=10, pin_memory=True)
    def get_data(loader):
        while True:
            yield from loader

    data = get_data(data)
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="datasets/speech_commands/train",
        save_dir="_checkpoints",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # # model flags
    # defaults['image_size'] = 32
    # defaults['num_channels'] = 128
    # defaults['num_res_blocks'] = 3
    # defaults['learn_sigma'] = False
    # defaults['dropout'] = 0.3
    # # diffusion flags
    # defaults['diffusion_steps'] = 200
    # defaults['noise_schedule'] = 'linear'
    # # train_flags
    # defaults['lr'] = 1e-4
    # defaults['batch_size'] = 64
    
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
