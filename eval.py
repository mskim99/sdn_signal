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

    loss_mse = []
    loss_ssim = []
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
            loss_mse.append(F.mse_loss(recon_images, norm_images).item())
            ssim_loss = metric.pytorch_ssim.SSIM(window_size=11)
            loss_ssim.append((1 - ssim_loss(recon_images, norm_images)).item())

    print("loss_ssim = ", round(sum(loss_ssim) / len(loss_ssim), 3))
    print("loss_mse = ", round(sum(loss_mse) / len(loss_mse), 3))

    if not os.path.exists("./sample/" + args.dataset_name + '/' + args.model):
        os.makedirs("./sample/" + args.dataset_name + '/' + args.model)
    sample_path = "./sample/" + args.dataset_name + '/' + args.model
    if args.model == 'snn-vae':
        sampled_x, sampled_z = model.sample(batch_size)
        functional.reset_net(model)
        recon_images = np.array(np.clip((sampled_x + 0.5).detach().cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
        recon_images = recon_images.reshape(4, 8, 28, 28)

        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        gs = fig.add_gridspec(4, 8)
        for n_row in range(4):
            for n_col in range(8):
                f_ax = fig.add_subplot(gs[n_row, n_col])
                f_ax.imshow(recon_images[n_row, n_col], cmap="gray")
                f_ax.axis("off")

        plt.savefig(sample_path + '/image.png')
        plt.show()

        all_images_list = []
        for i in range(40):
            sampled_x, sampled_z = model.sample(batch_size)
            recon_images = np.array(np.clip((sampled_x + 0.5).detach().cpu().numpy(), 0., 1.) * 255, dtype=np.uint8)
            if i == 0:
                all_images = recon_images
            else:
                all_images = np.concatenate((all_images, recon_images), axis=0)

        all_images_list.append(all_images)

    elif args.model == ('snn-vq-vae' or 'snn-vq-vae-uni'):
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
                functional.reset_net(model)
                functional.reset_net(denoise_fn)
                fig = plt.figure(figsize=(10, 5), constrained_layout=True)
                gs = fig.add_gridspec(4, 4)
                for n_row in range(4):
                    for n_col in range(4):
                        f_ax = fig.add_subplot(gs[n_row, n_col])
                        f_ax.imshow(generated_samples[n_row, n_col], cmap="gray")
                        f_ax.axis("off")

                if not os.path.exists(sample_path + '/' + str(tem)):
                    os.makedirs(sample_path + '/' + str(tem))
                plt.savefig(sample_path + '/' + str(tem) + '/image_' + str(tem) + '_' + str(k) + '.png')
                plt.show()

        temp = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        all_images_list = []
        for tem in temp:
            for i in tqdm.tqdm(range(80), desc='Sampling_for_temp=' + str(tem), total=80):
                sample = (abdiff.sample(temp=tem, sample_steps=49)).reshape(16, 48, 48)
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
                functional.reset_net(model)
                functional.reset_net(denoise_fn)
                if i == 0:
                    all_images = generated_samples
                else:
                    all_images = np.concatenate((all_images, generated_samples), axis=0)
            all_images_list.append(all_images)

    if args.metric == None or args.metric == 'IS':
        print("********now we get IS*********")
        if args.model == 'snn-vae':
            print(all_images.shape)
            torch.save(all_images, 'svae.pt')
            Is, _ = inception_score(np.repeat(all_images, 3, axis=1) / 255, cuda=True, batch_size=32, resize=True,
                                    splits=4)
            print("IS = ", Is)
        else:
            IS_list = []
            print(all_images_list[7].shape)
            torch.save(all_images_list[7], 'diff.pt')
            for all_images in tqdm.tqdm(all_images_list, desc='Get IS', total=len(all_images_list)):
                Is, _ = inception_score(np.repeat(all_images, 3, axis=1) / 255, cuda=True, batch_size=32, resize=True,
                                        splits=4)
                IS_list.append(Is)

            print('temp = ', temp)
            print('IS = ', IS_list)

    if args.metric == None or args.metric == 'KID':
        print("********now we get KID*********")
        kid = KernelInceptionDistance()
        for i, (images, labels) in enumerate(test_loader):
            if i == 40:
                break
            if i == 0:
                real_images = np.repeat(images, 3, axis=1)
            else:
                real_image = np.repeat(images, 3, axis=1)
                real_images = np.concatenate((real_images, real_image), axis=0)

        if args.model == 'snn-vae':
            A = torch.from_numpy(real_images * 255).type(torch.uint8)
            B = torch.from_numpy(np.repeat(all_images, 3, axis=1)).type(torch.uint8)

            kid.update(A, real=True)
            kid.update(B, real=False)
            kid_mean, kid_std = kid.compute()
            print("KID = ", kid_mean.item())
        else:
            KID_list = []
            for all_images in tqdm.tqdm(all_images_list, desc='Get KID', total=len(all_images_list)):
                A = torch.from_numpy(real_images * 255).type(torch.uint8)
                B = torch.from_numpy(np.repeat(all_images, 3, axis=1)).type(torch.uint8)

                kid.update(A, real=True)
                kid.update(B, real=False)
                kid_mean, kid_std = kid.compute()
                KID_list.append(kid_mean.item())
                print(kid_mean.item())
            print('temp = ', temp)
            print('KID = ', KID_list)

    if args.metric == None or args.metric == 'FID':
        print("********now we get FID*********")
        Fid_list = []
        for all_images in tqdm.tqdm(all_images_list, desc='Get FID', total=len(all_images_list)):
            torch.cuda.empty_cache()
            up = nn.Upsample(size=(299, 299), mode='bilinear').type(torch.cuda.FloatTensor)
            all_images = up(torch.Tensor(all_images / 255).cuda(0)).cpu().numpy()
            all_images = np.transpose(all_images, (0, 2, 3, 1))
            all_images = np.repeat(all_images, 3, axis=3)
            for i, (images, labels) in enumerate(test_loader):

                if i == 40:
                    break

                if i == 0:
                    real_image = np.repeat(images, 3, axis=1)
                    real_image = up(real_image.cuda(0)).cpu().numpy()
                    real_images = np.transpose(real_image, (0, 2, 3, 1))
                else:
                    real_image = np.repeat(images, 3, axis=1)
                    real_image = up(real_image.cuda(0)).cpu().numpy()
                    real_image = np.transpose(real_image, (0, 2, 3, 1))
                    real_images = np.concatenate((real_images, real_image), axis=0)

            Fid = calculate_fid(all_images, real_images, use_multiprocessing=False, batch_size=4)
            print(Fid)
            Fid_list.append(Fid)
        print('Fid = ', Fid_list)







