import os
from argparse import ArgumentParser

from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

from dataset import TrajectoryDataset
from model.trainer import DiffusionTrainer, ETATrainer
from model.diffusion import DiffusionProcess
from model.predictor import *

parser = ArgumentParser()

parser.add_argument('--cuda', help='the index of the cuda device', type=int, default=0)

# Dataset arguments
parser.add_argument('-n', '--name', help='the name of the dataset', type=str, required=True)
parser.add_argument('-s', '--split', help='the number of x and y split', type=int, default=20)
parser.add_argument('--flat', action='store_true')

# Denoiser arguments
parser.add_argument('--denoiser', help='name of the denoiser', type=str, default='unet')
parser.add_argument('--condition', type=str, default='odt')

# Training arguments
parser.add_argument('-e', '--epoch', help='number of training epoches', type=int, default=200)
parser.add_argument('-b', '--batch', help='batch size', type=int, default=128)
parser.add_argument('-t', '--timestep', help='the total timesteps of diffusion process', type=int, default=1000)
parser.add_argument('-p', '--partial', default=1.0, type=float)

# Diffusion training arguments
parser.add_argument('--traindiff', help='whether to train the diffusion model', action='store_true')
parser.add_argument('--loaddiff', help='whether to load the diffusion model from cached file', action='store_true')
parser.add_argument('--loadepoch', help='the training epoch to load from, set to -1 for loading from the final model', default=-1)
parser.add_argument('--loadgen', help='whether to load generated images', action='store_true')

# Draw generated image arguments
parser.add_argument('--draw', help='whether to draw generated images', action='store_true')
parser.add_argument('--numimage', help='number of images to draw', type=int, default=128)

# ETA training arguments
parser.add_argument('--traineta', help='whether to train the ETA prediction model', action='store_true')
parser.add_argument('--predictor', help='name of the predictor', type=str, default='trans')
parser.add_argument('--predictorL', help='number of layers in the predictor', type=int, default=2)
parser.add_argument('--trainorigin', help='whether to use original images to train ETA predictor', action='store_true')
parser.add_argument('--valorigin', help='whether to use original images to evaluate ETA predictor', action='store_true')
parser.add_argument('--stop', help='number of early stopping epoch for ETA training', type=int, default=10)

# Predictor arguments
parser.add_argument('-d', '--dmodel', help='dimension of predictor models', type=int, default=128)
parser.add_argument('--rmst', action='store_false')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
if torch.cuda.is_available():
    device = 'cuda:0'
    small = False
else:
    device = 'cpu'
    small = True

# Loading dataset
dataset = TrajectoryDataset(args.name, split=args.split, partial=args.partial, small=small, flat=args.flat)
dataset.load_images()

# Create models.
denoiser = Unet(dim=args.split, channels=dataset.num_channel, init_dim=4, dim_mults=(1, 2, 4), condition=args.condition)
diffusion = DiffusionProcess(T=args.timestep, schedule_name='linear')
diffusion_trainer = DiffusionTrainer(diffusion=diffusion, denoiser=denoiser, dataset=dataset, lr=1e-3,
                                     batch_size=args.batch, loss_type='huber', device=device, num_epoch=args.epoch)

# Load or train the diffusion model.
if args.loaddiff:
    denoiser = diffusion_trainer.load_model(None if args.loadepoch == -1 else args.loadepoch)
    # diffusion_trainer.eval_generation()
elif args.traindiff:
    denoiser = diffusion_trainer.train()

# Draw generated images.
if args.draw:
    val_images, val_ODTs, _ = dataset.get_images(1)
    val_images, val_ODTs = shuffle(val_images, val_ODTs)
    val_num = args.numimage
    gen_steps = diffusion.p_sample_loop(denoiser, shape=(val_num, *(val_images.shape[1:])),
                                        y=torch.from_numpy(val_ODTs[:val_num]).float().to(device),
                                        display=True)
    gen_images = gen_steps[-1]

    num_channel = gen_images.shape[1]
    for i in tqdm(range(val_num), desc='Drawing generated images'):
        plt.figure(figsize=(num_channel / 2 * 5, 5))
        for c in range(num_channel):
            plt.subplot(2, num_channel, c + 1)
            plt.title(f'Generated channel {c + 1}')
            plt.imshow(gen_images[i][c])

            plt.subplot(2, num_channel, c + 1 + num_channel)
            plt.title(f'Real channel {c + 1}')
            plt.imshow(val_images[i][c])
        plt.savefig(os.path.join('data', 'images', f'{dataset.name}_%03d.png' % i), bbox_inches='tight')
        plt.close()

# Train ETA predictor.
if args.traineta:
    if args.loadgen:
        diffusion_trainer.load_generation()
    else:
        indices_to_gen = [] + ([] if args.trainorigin else [0]) + ([] if args.valorigin else [1, 2])
        if len(indices_to_gen) > 0:
            diffusion_trainer.save_generation(indices_to_gen)

    if args.predictor == 'unet':
        predictor = UNetPredictor(dim=args.split, in_channel=dataset.num_channel, init_channel=args.dmodel, dim_mults=(1, 2, 4))
    elif args.predictor == 'trans':
        predictor = TransformerPredictor(input_dim=dataset.num_channel, d_model=args.dmodel, num_head=8, num_layers=args.predictorL,
                                         num_grid=dataset.split ** 2, dropout=0.1, use_st=args.rmst, use_grid=False)
    else:
        raise NotImplementedError('Undefined predictor "' + args.predictor + '"')
    eta_trainer = ETATrainer(diffusion=diffusion, predictor=predictor, dataset=dataset,
                             gen_images=diffusion_trainer.gen_images,
                             lr=1e-3, batch_size=args.batch, num_epoch=args.epoch, device=device, early_stopping=args.stop,
                             train_origin=args.trainorigin, val_origin=args.valorigin)
    eta_trainer.train()
