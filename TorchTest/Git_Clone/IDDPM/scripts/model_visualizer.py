import sys

sys.path.append("../")

import argparse

import torch as th

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    NUM_CLASSES
)

from torchviz import make_dot
from torchinfo import summary
import torch.onnx


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    model.to(dist_util.dev())
    model.eval()

    dummy_data = th.zeros(16, 3, 32, 32).to(dist_util.dev())
    torch.onnx.export(model, dummy_data, "model.onnx")
    # th.save(model, './md.pt')
    # 여기에 모델 분석 작성, diffusion 구조도 작성
    # make_dot(y_pred, params=dict(model.named_parameters())

    # summary(model)

    # # torchsummary말고 viz쓰려면 샘플 뜨고 역추적해야함
    # model_kwargs = {}
    # if args.class_cond:
    #     classes = th.randint(
    #         low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
    #     )
    #     model_kwargs["y"] = classes
    # sample_fn = (
    #     diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    # )
    # sample = sample_fn(
    #     model,
    #     (args.batch_size, 3, args.image_size, args.image_size),
    #     clip_denoised=args.clip_denoised,
    #     model_kwargs=model_kwargs,
    # )
    #
    # sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    # sample = sample.permute(0, 2, 3, 1)
    # sample = sample.contiguous()

    # make_dot(model(th.zeros(16, 3, 32, 32).to(dist_util.dev())), params=dict(list(model.named_parameters()))).render(filename="model_tree", directory="./", format="png")
    # make_dot(sample, params=dict(list(model.named_parameters()))).render(filename="model_tree", directory="./", format="png")

    # viz 거르고 onnx netron 함 해보러간다
    # th.onnx.export(model, torch.zeros(') 'model.onnx')


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=64,  # 10000
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
