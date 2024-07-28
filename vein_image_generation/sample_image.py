import torch
from stylegan2_pytorch import ModelLoader
from PIL import Image
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default='./')
    parser.add_argument('--project-name', type=str, default='default')
    parser.add_argument('--ckpt-idx', type=int, default=150)
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--out-dir', type=str, default='./generated_samples')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_samples = args.num_samples
    out_dir = args.out_dir

    loader = ModelLoader(
        base_dir=args.base_dir,   # path to where you invoked the command line tool
        name=args.project_name,   # the project name, defaults to 'default'
        load_from=args.ckpt_idx
    )

    os.makedirs(out_dir, exist_ok=True)

    i = 0
    with torch.no_grad():
        loader.model.GAN.eval()
        while i < num_samples:
            noise   = torch.randn(1, 512).cuda() # noise
            styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  # pass through mapping network
            images  = loader.styles_to_images(styles) # call the generator on intermediate style vectors
            ndarr = images[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:,:,0]
            im = Image.fromarray(ndarr)
            im.save("%s/%07d.bmp" % (out_dir, i))
            i += 1
            if i % 100 == 0:
                print(f"{i} samples generated.")


if __name__ == '__main__':
    main()
