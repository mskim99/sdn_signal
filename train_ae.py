import torch.utils.data.dataloader
import argparse
from load_dataset_snn import *

from snn_model.vae_model import *
from snn_model.vae_model_1d import SNN_VQVAE_1D
from spikingjelly.activation_based import functional

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store', dest='checkpoint',
                        help='The path of checkpoint, if use checkpoint')
    parser.add_argument('--dataset_name', type=str, default='MNIST')
    parser.add_argument('--dir_name', type=str, default='result_res_32_4')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='vq-vae')
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--sample_model', type=str, default='pixelsnn')
    parser.add_argument('--epochs', type=int, default=401)
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--ready', type=str, default=None)
    parser.add_argument('--mask', type=str, default='codebook_size')
    parser.add_argument('--codebook_size', type=int, default=128)
    parser.add_argument('--resolution', type=int, default=32)
    args = parser.parse_args()

    setup_seed(args.seed)
    if not os.path.exists("./" + args.dir_name + "/" + args.dataset_name + '/' + args.model):
        os.makedirs("./" + args.dir_name + "/" + args.dataset_name + '/' + args.model)
    save_path = "./" + args.dir_name + "/" + args.dataset_name + '/' + args.model

    batch_size = 32
    embedding_dim = 16
    num_embeddings = args.codebook_size
    if args.dataset_name == 'MNIST':
        train_loader, test_loader = load_mnist(data_path=args.data_path, batch_size=batch_size)
        print("load data: MNIST!")
    elif args.dataset_name == 'KMNIST':
        train_loader, test_loader = load_KMNIST(data_path=args.data_path, batch_size=batch_size)
        print("load data: KMNIST!")
    elif args.dataset_name == 'FMNIST':
        train_loader, test_loader = load_fashionmnist(data_path=args.data_path, batch_size=batch_size)
        print("load data: FMNIST!")
    elif args.dataset_name == 'Letters':
        train_loader, test_loader = load_MNIST_Letters(data_path=args.data_path, batch_size=batch_size)
        print("load data: Letters!")
    elif args.dataset_name == 'SignalImage':
        train_loader, test_loader = load_signal_to_image(data_path=args.data_path, batch_size=batch_size)
        print("load data: Signal & Image!")
    elif args.dataset_name == 'SignalImageLabel':
        train_loader, test_loader = load_signal_to_image(data_path=args.data_path, batch_size=batch_size)
        print("load data: Signal & Image (Label)!")
    elif args.dataset_name == 'Signal':
        train_loader, test_loader = load_signal_1d(data_path=args.data_path, batch_size=batch_size)
        print("load data: Signal (1D)!")

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    if args.model == 'snn-vq-vae':
        model = SNN_VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net=model, step_mode='m')
    if args.model == 'snn-vq-vae_cond':
        model = SNN_VQVAE_COND(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net=model, step_mode='m')
    if args.model == 'snn-vq-vae_1d':
        model = SNN_VQVAE_1D(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net=model, step_mode='s')
    elif args.model == 'snn-vq-vae-uni':
        model = SNN_VQVAE_uni(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net=model, step_mode='m')
    elif args.model == 'snn-vae':
        model = SNN_VAE()
        functional.set_step_mode(net=model, step_mode='m')
    elif args.model == 'vq-vae':
        model = VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
    model = model.cuda(0)
    print("The model is ready!")

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-5,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.001)

    # train VQ-VAE
    epochs = args.epochs
    print_freq = 20

    for epoch in range(epochs):
        model.train()

        print("Start training epoch {}".format(epoch, ))
        if args.model == 'vae':
            model.update_p(epoch, epochs)
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(0)
            labels = labels.int().cuda(0)
            # print(images.shape)
            if args.model != 'snn-vq-vae_1d':
                images = images - 0.5  # normalize to [-0.5, 0.5]
                images_spike = images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
            else:
                images = images - 0.5  # normalize to [-0.5, 0.5]
                # print(images.shape)
                images_spike = images.unsqueeze(0).repeat(16, 1, 1)
            if args.model == 'snn-vae':
                loss_eq, loss_rec = model(images_spike, images)
            elif args.model == 'snn-vq-vae' or args.model == 'snn-vq-vae-uni' or args.model == 'snn-vq-vae_1d':
                loss_eq, loss_rec, real_loss_rec = model(images_spike, images)
            elif args.model == 'snn-vq-vae_cond':
                loss_eq, loss_rec, real_loss_rec = model(images_spike, images, labels)
            elif args.model == 'vq-vae':
                loss_eq, loss_rec, real_loss_rec = model(images)
            optimizer.zero_grad()
            (loss_eq + loss_rec).backward()
            optimizer.step()

            if args.model != 'vq-vae':
                functional.reset_net(model)

            if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                if args.model == 'snn-vae':
                    print("[{}/{}][{}/{}]: loss {:.3f} loss_eq {:.3f} loss_rec {:.3f}".format(epoch, epochs, i,
                                                                                              len(train_loader),
                                                                                              (
                                                                                                      loss_eq + loss_rec).item(),
                                                                                              float(loss_eq),
                                                                                              float(loss_rec)))
                    # break
                else:
                    print("[{}/{}][{}/{}]: loss {:.3f} loss_eq {:.3f} loss_rec {:.3f}".format(epoch, epochs, i,
                                                                                              len(train_loader),
                                                                                              (
                                                                                                      loss_eq + loss_rec).item(),
                                                                                              float(loss_eq),
                                                                                              float(
                                                                                                  real_loss_rec)))
            # break
        # reconstructe images
        test_loader_iter = iter(test_loader)
        images, labels = next(test_loader_iter)

        n_samples = 32
        images = images[:n_samples]

        model.eval()

        norm_images = (images - 0.5).cuda(0)
        labels = labels.int().cuda(0)
        with torch.inference_mode():
            if args.model == 'vq-vae':
                e, recon_images = model(norm_images)
            elif args.model == 'snn-vae':
                images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                e, recon_images = model(images_spike, norm_images)
                functional.reset_net(model)
            elif args.model == 'snn-vq-vae':
                images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                # e, recon_images, _ = model(images_spike, norm_images)
                recon_images = model(images_spike, norm_images)
                functional.reset_net(model)
            elif args.model == 'snn-vq-vae_cond':
                images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                recon_images = model(images_spike, norm_images, labels)
                functional.reset_net(model)
            elif args.model == "snn-vq-vae_1d":
                images_spike = norm_images.repeat(16, 1, 1)
                e, recon_images, _ = model(images_spike, norm_images)
                functional.reset_net(model)
            elif args.model == "snn-vq-vae-uni":
                images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
                e, recon_images, _ = model(images_spike, norm_images)
                functional.reset_net(model)

        if epoch % 20 == 0:
            if args.model != 'snn-vq-vae_1d':
                recon_images = np.array(np.clip((recon_images + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
                ori_images = np.array(images.numpy() * 255, dtype=np.uint8)

                recon_images = recon_images.reshape(4, 8, args.resolution, args.resolution)
                ori_images = ori_images.reshape(4, 8, args.resolution, args.resolution)

                fig = plt.figure(figsize=(10, 10), constrained_layout=True)
                gs = fig.add_gridspec(8, 8)
                for n_row in range(4):
                    for n_col in range(8):
                        f_ax = fig.add_subplot(gs[n_row * 2, n_col])
                        f_ax.imshow(ori_images[n_row, n_col], cmap="gray")
                        f_ax.axis("off")
                        f_ax = fig.add_subplot(gs[n_row * 2 + 1, n_col])
                        f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
                        f_ax.axis("off")

                plt.savefig("./" + args.dir_name + "/" + args.dataset_name + '/' + args.model + "/epoch=" + str(epoch) + "_test.png")
            else:
                # print(recon_images.shape)
                np.save("./" + args.dir_name + "/" + args.dataset_name + '/' + args.model + "/epoch=" + str(epoch) + "_test_gt.npy", images.cpu().numpy())
                np.save("./" + args.dir_name + "/" + args.dataset_name + '/' + args.model + "/epoch=" + str(epoch) + "_test.npy", recon_images.cpu().numpy())

            torch.save(model.state_dict(), save_path + '/model_ae_' + str(epoch) + '.pth')
