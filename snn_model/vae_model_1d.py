import torch.utils.data.dataloader

from spikingjelly.activation_based import neuron, layer, surrogate

from .snn_layers import *


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.memout = MembraneOutputLayer()
        self.num_step = 16
        self.psp = PSP()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.alpha=1
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.poisson = nn.Sequential(
            layer.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
            layer.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        # (T,N,C,H,W)->(N,C,H,W) 16 128 16 7 7
        x_memout = (1 - self.alpha) * self.memout(x) + self.alpha * torch.sum(x, dim=0) / self.num_step
        # [128, 16, 7, 7] -> [128, 7, 7,16]
        x_memout = x_memout.permute(0, 2, 3, 1).contiguous()
        # [128, 7, 7,16] -> [6272, 16]
        flat_x = x_memout.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)

        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x_memout)  # [128, 7, 7, 16]

        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [128, 16, 7, 7]
            quantized = torch.unsqueeze(quantized, dim=0)  # [1, 128, 16, 7, 7]
            quantized = quantized.repeat(16, 1, 1, 1, 1)  # [16, 128, 16, 7, 7]
            quantized = self.poisson(quantized)  # [16, 128, 16, 7, 7]
            return quantized, encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x_memout.detach())

        # commitment loss
        e_latent_loss = F.mse_loss(x_memout, quantized.detach())

        loss_1 = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        # forward quantized = quantized
        # backward quantized = x
        quantized = x_memout + (quantized - x_memout).detach()
        # [128, 7, 7,16]->[128,16,7,7]
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # FIXME:
        quantized = torch.unsqueeze(quantized, dim=0)  # [1,128,16,7,7]
        quantized = quantized.repeat(16, 1, 1, 1, 1)  # [16,128,16,7,7]
        quantized = self.poisson(quantized)  # [16,128,16,7,7]

        q_latent_loss_2 = torch.mean((self.psp(quantized) - self.psp(x.detach())) ** 2)
        e_latent_loss_2 = torch.mean((self.psp(quantized.detach()) - self.psp(x)) ** 2)
        loss_2 = q_latent_loss_2 + self.commitment_cost * e_latent_loss_2

        return quantized, loss_1 + loss_2

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings.weight ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices)


class Encoder(nn.Module):
    """Encoder of VQ-VAE"""

    def __init__(self, in_dim=1, latent_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        self.snn_convs = nn.Sequential(
            layer.Conv1d(in_channels=in_dim, out_channels=32, kernel_size=3,
                         stride=2, padding=1),
            layer.BatchNorm1d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.Conv1d(in_channels=32, out_channels=64, kernel_size=3,
                         stride=2, padding=1),
            layer.BatchNorm1d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.Conv1d(in_channels=64, out_channels=latent_dim, kernel_size=1,
                         stride=1, padding=0),
            layer.BatchNorm1d(latent_dim),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        # [t, b, c, h, w]
        x = self.snn_convs(x)
        return x


class Decoder(nn.Module):
    """Decoder of VQ-VAE"""

    def __init__(self, out_dim=1, latent_dim=16):
        super().__init__()
        self.out_dim = out_dim
        self.latent_dim = latent_dim

        self.snn_convs = nn.Sequential(
            layer.ConvTranspose1d(in_channels=latent_dim, out_channels=64, kernel_size=3,
                                  stride=2, padding=1, output_padding=1),
            layer.BatchNorm1d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3,
                                  stride=2, padding=1, output_padding=1),
            layer.BatchNorm1d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),

            layer.ConvTranspose1d(in_channels=32, out_channels=out_dim, kernel_size=3,
                                  stride=1, padding=1, output_padding=0),

        )

    def forward(self, x):
        # [t, b, c, h, w]
        x = self.snn_convs(x)
        return x


class SNN_VQVAE_1D(nn.Module):
    """VQ-VAE"""

    def __init__(self, in_dim, embedding_dim, num_embeddings, data_variance,
                 commitment_cost=0.25):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.data_variance = data_variance

        self.encoder = Encoder(in_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)
        self.decoder = Decoder(in_dim, embedding_dim)
        # TODO:
        self.memout = MembraneOutputLayer()

    def forward(self, x, image):
        # x: [t, B, C, H, W]
        z = self.encoder(x)
        # print(z.shape)
        if not self.training:
            e, enco = self.vq_layer(z)
            x_recon = self.decoder(e)
            x_recon = torch.tanh(self.memout(x_recon))
            return e, x_recon, enco

        e, e_q_loss = self.vq_layer(z)
        x_recon = self.decoder(e)
        x_recon = torch.tanh(self.memout(x_recon))

        real_recon_loss = F.mse_loss(x_recon, image)
        recon_loss = real_recon_loss / self.data_variance

        return e_q_loss, recon_loss, real_recon_loss