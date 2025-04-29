import torch.utils.data.dataloader
import tqdm
import argparse
from snn_model.vq_diffusion import *
from load_dataset_snn import *
import metric.pytorch_ssim

from metric.IS_score import *
from metric.Fid_score import *
from torchmetrics.image.kid import KernelInceptionDistance

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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='vq-vae')
    parser.add_argument('--data_path', type=str, default='datasets')
    parser.add_argument('--sample_model', type=str, default='pixelsnn')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--ready', type=str, default=None)
    parser.add_argument('--mask', type=str, default='codebook_size')
    parser.add_argument('--codebook_size', type=int, default=128)
    args = parser.parse_args()

    setup_seed(args.seed)
    if not os.path.exists("./result/" + args.dataset_name + '/' + args.model):
        os.makedirs("./result/" + args.dataset_name + '/' + args.model)
    save_path = "./result/" + args.dataset_name + '/' + args.model

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

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    if args.model == 'snn-vq-vae':
        model = SNN_VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net=model, step_mode='m')
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

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.001)

    # train VQ-VAE
    epochs = args.epochs
    print_freq = 20

    # train diffusion
    model.load_state_dict(torch.load(save_path + '/model.pth'))
    train_indices = get_data_for_diff(train_loader, model)
    print(len(train_indices))
    print(train_indices[0].shape)
    print(train_indices[0][0])
    if args.mask == 'codebook_size':
        mask_id = num_embeddings
    elif args.mask == 'max':
        most_common_value, count = torch.mode(torch.flatten(train_indices[0]))
        mask_id = most_common_value
    elif args.mask == 'min':
        values, counts = torch.unique(torch.flatten(train_indices[0]), return_counts=True)
        least_common_value = values[torch.min(counts)].item()
        count = torch.min(counts).item()
        mask_id = count

    print("mask_id = ", mask_id)
    print('data for train diffusion is ready!')

    denoise_fn = DummyModel(1, num_embeddings).cuda(0)
    functional.set_step_mode(net=denoise_fn, step_mode='m')
    abdiff = AbsorbingDiffusion(denoise_fn, mask_id=mask_id)

    if not os.path.exists("./result/" + args.dataset_name + '/' + args.model + '/diff_result'):
        os.makedirs("./result/" + args.dataset_name + '/' + args.model + '/diff_result')
    save_path = "./result/" + args.dataset_name + '/' + args.model + '/diff_result'

    # optimizer = torch.optim.Adam(denoise_fn.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(denoise_fn.parameters(),
                                  lr=1e-3,
                                  betas=(0.9, 0.999),
                                  weight_decay=0.001)

    for epoch in range(epochs):

        denoise_fn.train()
        for batch_idx, (indices) in enumerate(train_indices):
            indices = indices.float().cuda(0)
            indices = indices.unsqueeze(dim=1)
            loss = abdiff.train_iter(indices)
            loss = loss['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net=denoise_fn)
            print("[{}/{}][{}/{}]: loss {:.3f} ".format(epoch, epochs, batch_idx, len(train_loader),
                                                        (loss).item()))
            # break
        if epoch % 10 == 0:

            denoise_fn.eval()
            sample_list = []
            for i in range(2):
                sample = (abdiff.sample(temp=0.65, sample_steps=49)).reshape(16, 7, 7)
                sample_list.append(sample)
            sample = torch.cat(sample_list, dim=0)
            with torch.inference_mode():
                z = model.vq_layer.quantize(sample.cuda(0))

                z = z.permute(0, 3, 1, 2).contiguous()
                quantized = torch.unsqueeze(z, dim=0)
                quantized = quantized.repeat(16, 1, 1, 1, 1)
                quantized = model.vq_layer.poisson(quantized)

                pred = model.decoder(quantized)
                pred = torch.tanh(model.memout(pred))

            generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
            generated_samples = generated_samples.reshape(4, 8, 28, 28)

            fig = plt.figure(figsize=(10, 5), constrained_layout=True)
            gs = fig.add_gridspec(4, 8)
            for n_row in range(4):
                for n_col in range(8):
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    f_ax.imshow(generated_samples[n_row, n_col], cmap="gray")
                    f_ax.axis("off")
            plt.savefig(save_path + "/epoch=" + str(epoch) + "_test.png")
            torch.save(denoise_fn.state_dict(), save_path + '/diff_model.pth')