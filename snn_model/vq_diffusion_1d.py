import math

import torch.utils.data.dataloader
import torch.distributions as dists

from spikingjelly.activation_based import functional

from .vae_model import *

def get_data_for_diff_1d(train_loader, model):
    print('prepare data for train diffusion...')
    model.eval()
    train_indices = []
    for images, labels in train_loader:
        # images = images - 0.5 # normalize to [-0.5, 0.5]
        images = images.cuda()
        images_spike = images.repeat(16, 1, 1)

        with torch.inference_mode():
            _, _, encoding_indices = model(images_spike, images)
            train_indices.append(encoding_indices.reshape(images.shape[0],-1).cpu())

    return train_indices


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        pass


class AbsorbingDiffusion1D(Sampler):
    def __init__(self, denoise_fn, mask_id):
        super().__init__()
        # self.num_classes = denoise_fn.num_embeddings
        self.num_classes = mask_id
        self.shape = [7, 7]
        self.num_timesteps = 49
        self.mask_id = mask_id
        self._denoise_fn = denoise_fn
        self.n_samples = 16
        self.mask_schedule = 'random'
        self.loss_type = 'mse'

    def sample_time(self, b, device):
        t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        pt = torch.ones_like(t).float() / self.num_timesteps
        return t, pt

    def q_sample(self, x_0, t):
        x_t, x_0_ignore = x_0.clone(), x_0.clone()
        t_mask = t.reshape(x_0.shape[0], 1, 1)
        t_mask = t_mask.expand(x_0.shape[0], 1, 128)
        mask = torch.rand_like(x_t.float()) < (t_mask.float() / self.num_timesteps)

        x_t[mask] = self.mask_id

        x_0_ignore[torch.bitwise_not(mask)] = -1
        return x_t, x_0_ignore, mask

    def _train_loss(self, x_0):

        b, device = x_0.size(0), x_0.device

        t, pt = self.sample_time(b, device)

        x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)

        x_0_hat_logits = self._denoise_fn(x_t, t=t)

        if self.loss_type == 'mse':
            loss = torch.abs(x_0_hat_logits - x_0_ignore)
        elif self.loss_type == 'elbo' or self.loss_type == 'reweighted_elbo':
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)

            x_0_hat_logits = self._denoise_fn(x_t, t=t)

            cross_entropy_loss = F.cross_entropy(x_0_hat_logits.reshape(b, self.num_classes, 49),
                                                 x_0_ignore.reshape(b, 49).type(torch.LongTensor).to(device),
                                                 ignore_index=-1, reduction='none').sum(1)

            vb_loss = cross_entropy_loss / t
            vb_loss = vb_loss / pt
            vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())

            if self.loss_type == 'elbo':
                loss = vb_loss
            elif self.loss_type == 'reweighted_elbo':
                weight = (1 - (t / self.num_timesteps))
                loss = weight * cross_entropy_loss
                loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        return loss.mean()

    def sample(self, temp=1.0, sample_steps=None):

        b, device = int(self.n_samples), 'cuda'
        x_t = torch.ones(b, 1, 256, device=device).long() * self.mask_id
        # x_t = torch.ones(b, 1, 48, 48, device=device).long() * self.mask_id
        unmasked = torch.zeros_like(x_t, device=device).bool()

        sample_steps = list(range(1, sample_steps + 1))
        for t in reversed(sample_steps):
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            t_mask = t.reshape(b, 1, 1)
            t_mask = t_mask.expand(b, 1, 256)
            # t_mask = t_mask.expand(b, 1, 48, 48)
            changes = torch.rand_like(x_t.float()) < 1 / t_mask.float()
            changes = changes

            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # x_t_for_unet = F.pad(input=x_t, pad=(0, 1, 1, 0), mode='constant', value = mask_id) # for unet
            # denoise_fn = self._denoise_fn.cuda(1)
            x_0_logits = self._denoise_fn(x_t.float(), t=t).permute(0, 2, 1)
            functional.reset_net(self._denoise_fn)
            # x_0_logits = x_0_logits[:,:,1:,0:-1] # for unet

            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(
                logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0_hat = x_0_hat.unsqueeze(dim=1)
            x_t[changes] = x_0_hat[changes]

        return x_t

    def train_iter(self, x):
        loss = self._train_loss(x)
        stats = {'loss': loss}
        return stats