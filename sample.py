import torch.utils.data.dataloader
import tqdm
import argparse
from snn_model.vq_diffusion import *
from load_dataset_snn import *
import metric.pytorch_ssim

from metric.IS_score import *
from metric.Fid_score import *
from torchmetrics.image.kid import KernelInceptionDistance

import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--ready', type=str, default=None)
    parser.add_argument('--mask', type=str, default='codebook_size')
    parser.add_argument('--codebook_size', type=int, default=128)
    args = parser.parse_args()

    setup_seed(args.seed)
    if not os.path.exists("./result/" + args.dataset_name + '/' + args.model):
        os.makedirs("./result/" + args.dataset_name + '/' + args.model)
    save_path = "./result/" + args.dataset_name + '/' + args.model

    batch_size = 1
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

    # compute the variance of the whole training set to normalise the Mean Squared Error below.
    train_images = []
    for images, labels in train_loader:
        train_images.append(images)
    train_images = torch.cat(train_images, dim=0)
    train_data_variance = torch.var(train_images)

    if args.model == 'snn-vq-vae':
        model = SNN_VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'snn-vq-vae-uni':
        model = SNN_VQVAE_uni(1, embedding_dim, num_embeddings, train_data_variance)
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'snn-vae':
        model = SNN_VAE()
        functional.set_step_mode(net = model, step_mode = 'm')
    elif args.model == 'vq-vae':
        model = VQVAE(1, embedding_dim, num_embeddings, train_data_variance)
    model = model.cuda(0)

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

    denoise_fn = DummyModel(1, num_embeddings).cuda(0)
    functional.set_step_mode(net=denoise_fn, step_mode='m')
    abdiff = AbsorbingDiffusion(denoise_fn, mask_id=mask_id)

    print("The model is ready!")

    model.load_state_dict(torch.load(save_path + '/model.pth'))
    denoise_fn.load_state_dict(torch.load(save_path + '/diff_result/diff_model.pth'))

    model.eval()
    denoise_fn.eval()

    for i, (images, labels) in enumerate(test_loader):
        norm_images = (images - 0.5).cuda(0)
        with torch.inference_mode():
            images_spike = norm_images.unsqueeze(0).repeat(16, 1, 1, 1, 1)
            if args.model == 'vq-vae':
                e, recon_images = model(norm_images)
            elif args.model == ('snn-vae' or 'snn-vq-vae-uni'):
                e, recon_images = model(images_spike, norm_images)
                functional.reset_net(model)
            else:
                e, recon_images, _ = model(images_spike, norm_images)
                functional.reset_net(model)

    if not os.path.exists("./sample_part/" + args.dataset_name + '/' + args.model):
        os.makedirs("./sample_part/" + args.dataset_name + '/' + args.model)
    sample_path = "./sample_part/" + args.dataset_name + '/' + args.model

    denoise_fn.eval()
    temp = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for tem in tqdm.tqdm(temp, desc='drawing', total=len(temp)):
        for k in range(20):
            sample_list = []
            for i in range(1):
                sample = (abdiff.sample(temp=tem, sample_steps=100)).reshape(16, 48, 48)
                sample_list.append(sample)
                functional.reset_net(denoise_fn)
            sample = torch.cat(sample_list, dim=0)
            with torch.inference_mode():
                z = model.vq_layer.quantize(sample.cuda(0))

                z = z.permute(0, 3, 1, 2).contiguous()

                quantized = torch.unsqueeze(z, dim=0)
                quantized = quantized.repeat(16, 1, 1, 1, 1)
                quantized = model.vq_layer.poisson(quantized)
                # torch.Size([128, 16, 7, 7, 16])

                pred = model.decoder(quantized)
                pred = torch.tanh(model.memout(pred))

            generated_samples = np.array(np.clip((pred + 0.5).cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
            generated_samples = generated_samples.reshape(4, 4, 192, 192)
            cv2.imwrite(sample_path + '/' + str(tem) + '/image_' + str(tem) + '_' + str(k) + '.png', generated_samples[0, 0, :, :])
