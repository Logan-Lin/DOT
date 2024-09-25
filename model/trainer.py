import os
import math
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from torch import nn
from tqdm import tqdm
import pandas as pd

from model.diffusion import DiffusionProcess


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


def mean_absolute_percentage_error(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape


def cal_regression_metric(label, pres, p=True, save=False, save_path='undefined'):
    rmse = math.sqrt(mean_squared_error(label, pres))
    mae = mean_absolute_error(label, pres)
    mape = mean_absolute_percentage_error(label, pres)

    if p:
        print('rmse: %05.6f, mae: %05.6f, mape: %05.6f' % (rmse, mae, mape * 100), flush=True)

    if save:
        s = pd.Series([rmse, mae, mape], index=['rmse', 'mae', 'mape'])
        s.to_hdf(f'{save_path}.h5',
                 key=f't{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}', format='table')
        np.savez(f'{save_path}.npz',
                 pre=pres, label=label)
        print('[Saved prediction]')

    return rmse, mae, mape


class DiffusionTrainer:
    """
    Trainer for the diffusion model.
    """

    def __init__(self, diffusion: DiffusionProcess, denoiser, dataset, lr, batch_size, loss_type,
                 num_epoch, device):
        """
        :param diffusion: diffusion model for sampling from q and p.
        :param denoiser: reverse denoise diffusion model.
        :param dataset:
        :param lr:
        :param device:
        """
        self.diffusion = diffusion
        self.denoiser = denoiser.to(device)

        self.optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr)
        self.dataset = dataset

        self.batch_size = batch_size
        self.loss_type = loss_type
        self.num_epoch = num_epoch
        self.device = device

        self.save_model_path = os.path.join(dataset.meta_dir,
                                            f'denoiser_{self.denoiser.name}_'
                                            f'T{self.diffusion.T}_'
                                            f'S{self.dataset.split}.model')

        gen_path = os.path.join(dataset.meta_dir,
                                f'images_{self.denoiser.name}_'
                                f'T{self.diffusion.T}_'
                                f'S{self.dataset.split}')
        self.gen_set_path = gen_path + '.npz'

        self.gen_images = [np.array([]) for _ in range(3)]

    def train_epoch(self, meta):
        self.denoiser.train()
        losses = []
        batch_iter = list(zip(*meta))

        desc_txt = 'Training diffusion, loss %05.6f'
        with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
                  desc=desc_txt % 0.0) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                batch_img, batch_odt, _ = zip(*batch)
                # Create two batch tensors, with shape (N, C, X, Y) and (N, y_feat).
                batch_img, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                        for item in (batch_img, batch_odt))
                t = torch.randint(0, self.diffusion.T, (batch_img.size(0),)).long().to(self.device)

                loss = self.diffusion.p_losses(self.denoiser, batch_img, t, batch_odt, loss_type=self.loss_type)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_description(desc_txt % (loss.item()))
        return float(np.mean(losses))

    def eval_generation(self):
        meta = self.dataset.get_images(2)

        real_img, _, _ = meta
        generation = self.generate(meta)
        real_img = real_img.reshape(real_img.shape[0], -1)
        generation = generation.reshape(generation.shape[0], -1)
        cal_regression_metric(real_img, generation, save=True,
                              save_path=os.path.join(self.dataset.meta_dir, f'denoiser_{self.denoiser.name}_C{self.denoiser.condition}_T{self.diffusion.T}_'
                                        f'S{self.dataset.split}_P{self.dataset.partial}'))

    def train(self):
        train_meta = self.dataset.get_images(0)

        for epoch in range(self.num_epoch):
            train_loss = self.train_epoch(train_meta)
            # self.save_model(epoch)
        self.save_model()
        self.eval_generation()
        return self.denoiser

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        torch.save(self.denoiser.state_dict(), save_path)
        print('[Saved denoiser] to ' + save_path)

    def load_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        self.denoiser.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        print('[Loaded denoiser] from ' + save_path)
        return self.denoiser

    def generate(self, meta):
        self.denoiser.eval()

        batch_iter = list(zip(*meta))
        gens = []
        for batch in tqdm(next_batch(batch_iter, self.batch_size), total=len(batch_iter) // self.batch_size,
                          desc='Generating images'):
            batch_img, batch_odt, batch_arr = zip(*batch)
            batch_odt = torch.from_numpy(np.stack(batch_odt, 0)).float().to(self.device)

            gen = self.diffusion.p_sample_loop(self.denoiser, shape=np.array(batch_img).shape,
                                               y=batch_odt, display=False)[-1]
            gens.append(gen)
        return np.concatenate(gens, axis=0)

    def save_generation(self, select_sets=None):
        if select_sets is None:
            select_sets = range(3)
        for s in select_sets:
            self.gen_images[s] = self.generate(self.dataset.get_images(s))
        np.savez(self.gen_set_path, train=self.gen_images[0], val=self.gen_images[1], test=self.gen_images[2])
        print('[Saved generation] images to ' + self.gen_set_path)

    def load_generation(self):
        gen_images = np.load(self.gen_set_path)
        self.gen_images = [gen_images[label] for label in ['train', 'val', 'test']]
        print('[Loaded generation] from ' + self.gen_set_path)


class ETATrainer:
    """
    Trainer for the ETA predictor.
    """

    def __init__(self, diffusion, predictor, dataset, gen_images, lr, batch_size, num_epoch, device,
                 train_origin=False, val_origin=False, early_stopping=-1):
        self.diffusion = diffusion
        self.predictor = predictor.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.dataset = dataset
        self.gen_images = gen_images
        self.loss_func = nn.MSELoss()

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.device = device

        self.train_origin = train_origin
        self.val_origin = val_origin
        self.early_stopping = early_stopping

        self.save_model_path = os.path.join(dataset.meta_dir,
                                            f'predictor_{self.predictor.name}_'
                                            f'L{self.predictor.num_layers}_D{self.predictor.d_model}_'
                                            f'st{self.predictor.use_st}_grid{self.predictor.use_grid}_'
                                            f'T{self.diffusion.T}_'
                                            f'S{self.dataset.split}.model')

    def train_epoch(self, eta_input, meta):
        self.predictor.train()
        losses = []
        origin_image, odt, label = meta
        batch_iter = list(zip(origin_image if self.train_origin else eta_input, odt, label))

        desc_txt = 'Training eta predictor, loss %05.6f'
        with tqdm(next_batch(shuffle(batch_iter), self.batch_size), total=len(batch_iter) // self.batch_size,
                  desc=desc_txt % 0.0) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                batch_input, batch_odt, batch_label = zip(*batch)
                batch_input, batch_odt, batch_label = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                                       for item in (batch_input, batch_odt, batch_label))
                pre = self.predictor(batch_input, batch_odt)  # (batch_size)

                loss = self.loss_func(pre, batch_label)
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())
                pbar.set_description(desc_txt % (loss.item()))
        return float(np.mean(losses))

    def eval_epoch(self, eta_input, meta, save=False):
        self.predictor.eval()
        origin_image, odt, label = meta
        batch_iter = list(zip(origin_image if self.val_origin else eta_input, odt))
        pres = []

        for batch in next_batch(batch_iter, self.batch_size):
            batch_input, batch_odt = zip(*batch)
            batch_input, batch_odt = (torch.from_numpy(np.stack(item, 0)).float().to(self.device)
                                      for item in (batch_input, batch_odt))
            pre = self.predictor(batch_input, batch_odt).detach().cpu().numpy()  # (batch_size)
            pres.append(pre)

        pres = np.concatenate(pres)
        return cal_regression_metric(label, pres, save=save,
                                     save_path=os.path.join(self.dataset.meta_dir, f'{self.predictor.name}_L{self.predictor.num_layers}_D{self.predictor.d_model}_'
                                     f'st{self.predictor.use_st}_grid{self.predictor.use_grid}_'
                                     f'S{self.dataset.split}_P{self.dataset.partial}'))

    def train(self):
        train_gen = self.gen_images[0]
        train_meta = self.dataset.get_images(0)
        val_gen = self.gen_images[1]
        val_meta = self.dataset.get_images(1)
        test_gen = self.gen_images[2]
        test_meta = self.dataset.get_images(2)
        min_val_mae = 1e8
        epoch_before_stop = 0

        for epoch in range(self.num_epoch):
            loss_val = self.train_epoch(train_gen, train_meta)
            val_rmse, val_mae, val_mape = self.eval_epoch(val_gen, val_meta)
            self.save_model(epoch)

            if min_val_mae > val_mae:
                min_val_mae = val_mae
                epoch_before_stop = 0
            else:
                epoch_before_stop += 1

            if 0 < self.early_stopping <= epoch_before_stop:
                print('\nEarly stopping, best epoch:', epoch - epoch_before_stop)
                self.load_model(epoch - epoch_before_stop)
                break

        test_rmse, test_mae, test_mape = self.eval_epoch(test_gen, test_meta, save=True)
        self.save_model()
        return self.predictor

    def save_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Saved predictor] to ' + save_path)
        torch.save(self.predictor.state_dict(), save_path)

    def load_model(self, epoch=None):
        save_path = self.save_model_path
        if epoch is not None:
            save_path += f'_epoch{epoch}'
        print('[Loaded predictor] from ' + save_path)
        self.predictor.load_state_dict(torch.load(save_path, map_location=self.device), strict=True)
        self.predictor.eval()
        return self.predictor
