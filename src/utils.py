import os
from datetime import datetime
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from scipy import linalg
from torchaudio.functional import frechet_distance as frechet_distance_torchaudio


# image alterations
def apply_image_corruption(images, mode, random_pix_ind=None, background_removal_frac=None, noise_level=None):
    if mode == 'percentile':
        assert background_removal_frac is not None
        for i, image in enumerate(images):
            thresh = np.percentile(image.cpu(), background_removal_frac * 100)
            images[i][image <= thresh] = 0.

        # thresh = np.percentile(images.cpu(), background_removal_frac * 100)
        # images[images <= thresh] = 0.
    elif mode == 'lower_rows':
        frac = 0.01
        images[:, :, int(frac * images.size(2)):, :] = 0.
    elif mode == 'single_pixel':
        assert random_pix_ind is not None
        images[:, :, :random_pix_ind, :random_pix_ind] = 0.
        images[:, :, random_pix_ind+1:, random_pix_ind+1:] = 0.
    elif mode == 'noise':
        assert noise_level is not None
        images = images + np.random.randn(*images.shape) * noise_level
    elif mode == "smoothing":
        assert noise_level is not None
        is_tensor = type(images) == torch.Tensor
        if is_tensor:
            images = images.cpu().numpy() 

        for i, image in enumerate(images):
            images[i] = gaussian_filter(image, sigma=noise_level)

        if is_tensor:
            images = torch.from_numpy(images).float().cuda()
    else:
        raise NotImplementedError
    return images

# logging
class Logger():
    def __init__(self, mode, log_dir, custom_name=''):
        assert mode in ['custom']
        self.mode = mode
        
        # create log file
        now = datetime.now()
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        logfname = 'log_{}_{}.txt'.format(custom_name, now.strftime("%m-%d-%Y_%H:%M:%S"))
        self.logfname = os.path.join(log_dir, logfname)
        print(self.logfname)
        
        with open(self.logfname, 'w') as fp: # create file
            pass
        
        # log intro message
        start_msg = 'beginning {} on {}.\n'.format(self.mode, now.strftime("%m/%d/%Y, %H:%M:%S"))

            
        if mode == 'custom':
            start_msg += '--------------------------\n'
            start_msg += 'custom log.\n'
        
        self.write_msg(start_msg)
        print(start_msg)
        
    def write_msg(self, msg, print_=True):
        if print_:
            print(msg)
            
        if not msg.endswith('\n'):
            msg += '\n'
            
        log_f = open(self.logfname, 'a')
        log_f.write(msg)
        log_f.close()
        
        return
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_netinput_onechannel(net, model):
    # fix nets to take one channel as input
    name = model.__name__
    if 'resnet' in name:
        net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif 'vgg' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    elif 'squeezenet' in name:
        net.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))
    elif 'densenet' in name:
        net.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif 'swin' in name:
        net.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    elif 'TinyConv' in name:
        net.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    else:
        raise NotImplementedError
        
def change_num_output_features(net, model, num_output_features):
    # fix nets to take one channel as input
    name = model.__name__
    if 'resnet' in name:
        net.fc = nn.Linear(512, out_features=num_output_features, bias=True)
    elif 'swin' in name:
        net.head = nn.Linear(768, out_features=num_output_features, bias=True)
    elif 'TinyConv' in name:
        net.lin = nn.Linear(224*224*64, out_features=num_output_features, bias=True)
    else:
        raise NotImplementedError

def get_layer(net, model, layer_name):
    name = model.__name__
    if 'resnet' in name:
        #return net.layer4[-1], 'layer4.1'
        if layer_name == "layer1":
            return net.layer1
        elif layer_name == "layer2":
            return net.layer2
        elif layer_name == "layer3":
            return net.layer3
        elif layer_name == "layer4":
            return net.layer4
        elif layer_name == "layer1.0":
            return net.layer1[0]
        elif layer_name == "conv1":
            return net.conv1
        elif layer_name == "relu":
            return net.relu
        else:
            raise NotImplementedError
    elif 'UNet' in name:
        if layer_name.startswith("encoder_encoding_blocks_") and layer_name.endswith("_conv2"):
            idx = int(layer_name.split("_")[3])
            return net.encoder.encoding_blocks[idx].conv2
        elif layer_name == "encoder_encoding_blocks_0_conv1_activation_layer":
            return net.encoder.encoding_blocks[0].conv1.activation_layer
        elif layer_name == "encoder":
            return net.encoder
        elif layer_name == "bottom_block":
            return net.bottom_block
        elif layer_name == "decoder":
            return net.decoder
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

# metrics
def get_auc_score(all_gts, all_preds):
    try:
        auc = roc_auc_score(all_gts, all_preds)
    except ValueError:
        auc = np.nan
    return auc

def dice_coeff(pred, target):
    # dice coefficient for flattened masks of shape d

    eps = 0.0001
    inter = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) + eps

    t = (2 * inter.float() + eps) / union.float()
    return t

# adapted from gan-metrics-pytorch
def frechet_distance(feats1, feats2, eps=1e-6, means_only=False):
    # feats1 and feats2 are N1 x M and N2 x M matrices
    m1 = np.mean(feats1, axis=0)
    s1 = np.cov(feats1, rowvar=False)
    m2 = np.mean(feats2, axis=0)
    s2 = np.cov(feats2, rowvar=False)

    mu1 = np.atleast_1d(m1)
    mu2 = np.atleast_1d(m2)

    sigma1 = np.atleast_2d(s1)
    sigma2 = np.atleast_2d(s2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    if means_only:
        return diff.dot(diff)

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def frechet_distance_torch(feats1, feats2):
    # feats1 and feats2 are N1 x M and N2 x M matrices
    m1 = torch.mean(feats1, dim=0)
    s1 = torch.cov(feats1)
    m2 = torch.mean(feats2, dim=0)
    s2 = torch.cov(feats2)

    mu1 = torch.atleast_1d(m1)
    mu2 = torch.atleast_1d(m2)

    sigma1 = torch.atleast_2d(s1)
    sigma2 = torch.atleast_2d(s2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    fd = frechet_distance_torchaudio(mu1, sigma1, mu2, sigma2)

    return fd

# from https://github.com/yiftachbeer/mmd_loss_pytorch?tab=readme-ov-file
class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device='cuda'):
        super().__init__()
        self.device = device

        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(self.device))[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

# compute cosine similarity between two torch vectors
def cosine_similarity(a, b):
    a = a.float()
    b = b.float()
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

def contrastive_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate embeddings
    z = torch.cat((z_i, z_j), dim=0)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T)

    # Labels for contrastive loss
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    
    # Remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    # Apply temperature
    sim_matrix /= temperature

    # Compute contrastive loss
    loss = -torch.log(torch.exp(sim_matrix) / torch.sum(torch.exp(sim_matrix), dim=1, keepdim=True))
    loss = loss * labels
    loss = loss.sum() / (2 * batch_size)
    
    return loss
