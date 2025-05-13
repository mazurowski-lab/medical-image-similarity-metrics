import os
import sys
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import pytorch_gan_metrics as metrics
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
from argparse import ArgumentParser

# set random seed
fix_seed = True
if fix_seed:
    seed = 1338
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(
        metric_name="swd",
        dset_path='../data/brats/testA',
        samples_path='../data/brats/harmonized/t1tot2/by_cut',

        batch_size=16,
        img_size=256,
        compute_new_FID_stats=True
):

    if metric_name in ['fid', 'is', 'swd']:
        # create datasets using pytorch image folder
        samples_dataset = metrics.ImageDataset(
            samples_path, 
            exts=['png', 'jpg'],
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            )

        dataset = metrics.ImageDataset(
            dset_path, 
            exts=['png', 'jpg'],
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
            )

        # if datasets are of different size, use random sampling to get the same size
        if len(samples_dataset) != len(dataset):
            import random
            if len(samples_dataset) > len(dataset):
                print("samples set too large,")
                samples_dataset = torch.utils.data.Subset(samples_dataset, random.sample(range(len(samples_dataset)), len(dataset)))
            else:
                print("real set too large,")
                dataset = torch.utils.data.Subset(dataset, random.sample(range(len(dataset)), len(samples_dataset)))

            print("NOTICE: datasets are of different size, using random sampling to get the same size")

        samples_loader = DataLoader(samples_dataset, batch_size=50, num_workers=4)
        loader = DataLoader(dataset, batch_size=50, num_workers=4)

    if metric_name in ['fid', 'is']:
        # FID and IS
        stats_path = os.path.join("stats/fid", "{}.npz".format(os.path.basename(dset_path)))
        if compute_new_FID_stats:
            metrics.utils.calc_and_save_stats(
                dset_path,
                stats_path,
                batch_size,
                img_size
            )

        (IS, IS_std), FID = metrics.get_inception_score_and_fid(
        samples_loader, stats_path)
        print("IS: {} +- {}".format(IS, IS_std))
        print("FID: {}".format(FID))
    elif metric_name == 'swd':
        sys.path.append('swd-pytorch')
        from swd import swd

        # create datasets


        # convert datasets to N x 3 x H x W tensors
        x1 = torch.cat([s.unsqueeze(0) for s in samples_dataset])
        x2 = torch.cat([s.unsqueeze(0) for s in dataset])

        out = swd(x1, x2, device="cuda") # Fast estimation if device="cuda"
        print("swd: {}".format(out.item())) # tensor(53.6950)
    elif metric_name == 'ssim+mse':
        ssims = []
        mses = []
        for img_fname in tqdm(os.listdir(dset_path)):
            generated_img_path = os.path.join(samples_path, "condon_" + img_fname)
            if not os.path.exists(generated_img_path):
                og_path = generated_img_path
                generated_img_path = os.path.join(os.path.split(og_path)[0], os.path.split(og_path)[1].replace("condon_", "controlnet_"))
                if not os.path.exists(generated_img_path):
                    generated_img_path = os.path.join(samples_path, img_fname)

            img2 = np.array(Image.open(generated_img_path).convert('L'))
            img1 = np.array(Image.open(os.path.join(dset_path, img_fname)).resize(img2.shape).convert('L'))

            ssim_val = ssim(img1, img2)
            ssims.append(ssim_val)

            mse_val = np.mean((img1 - img2) ** 2)
            mses.append(mse_val)

        print("SSIM: {}".format(np.mean(ssims)))
        print("MSE: {}".format(np.mean(mses)))
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--metric_name", type=str)
    parser.add_argument("--true", type=str)
    parser.add_argument("--fake", type=str)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)

    args = parser.parse_args()


    main(
        metric_name=args.metric_name,
        dset_path=args.true,
        samples_path=args.fake,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
