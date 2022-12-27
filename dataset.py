import os
import math
from random import choices
from collections import Counter
from argparse import ArgumentParser

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm

pd.options.mode.chained_assignment = None


class TrajectoryDataset:
    """
    Dataset class for trajectories.
    """

    def __init__(self, name, split, partial=1.0, small=False, flat=False):
        """
        :param file_path:
        """
        self.split = split
        self.name = name
        self.partial = partial
        self.file_path = os.path.join('data', self.name)
        self.flat = flat

        self.image_meta_path = os.path.join('/nfs/srv/data1/yanlin/DiffusionETA/meta', 
                                            self.name + f'_S{self.split}_F{self.flat}') + '_image.npz'
        self.traj_meta_path = os.path.join('/nfs/srv/data1/yanlin/DiffusionETA/meta', 
                                           self.name + f'_S{self.split}') + '_traj.npz'

        # A DataFrame containing the full dataset.
        # This DataFrame should contain four columns: trip_id, lng, lat, time.
        if small:
            self.dataframe = pd.read_hdf(os.path.join('data', name) + '_small.h5', key='raw')
        else:
            file_path = '/nfs/srv/data1/yanlin/Dataset/DeepGTT/' + name + '.h5'
            self.dataframe = pd.read_hdf(file_path, key='raw')
            self.dataframe = self.dataframe[self.dataframe['trip_id'].isin(pd.read_hdf(file_path, key='valid'))]
            print(f"[Loaded dataset] {file_path}, number of travels {self.dataframe['trip_id'].drop_duplicates().shape[0]}")

        trip_len_counter = Counter(self.dataframe['trip_id'])
        self.max_len = max(trip_len_counter.values())
        num_trips = len(trip_len_counter)
        all_trips = list(trip_len_counter.keys())

        # Transform time into the time of the day.
        self.dataframe['daytime'] = (self.dataframe['time'] % (24 * 60 * 60))

        # Normalize lng, lat, time into [0, 1] values.
        self.col_minmax = {}
        for col in ['lng', 'lat', 'time']:
            self.col_minmax[col + '_min'] = self.dataframe[col].min()
            self.col_minmax[col + '_max'] = self.dataframe[col].max()
            self.dataframe[col + '_norm'] = (self.dataframe[col] - self.dataframe[col].min()) / \
                                            (self.dataframe[col].max() - self.dataframe[col].min())
            self.dataframe[col + '_norm'] = self.dataframe[col + '_norm'] * 2 - 1
        self.dataframe['daytime_norm'] = self.dataframe['daytime'] / 60 / 60 / 24 * 2 - 1

        x_index = np.around((self.dataframe['lng'] - self.col_minmax['lng_min']) /
                            ((self.col_minmax['lng_max'] - self.col_minmax['lng_min']) / (self.split - 1)))
        y_index = np.around((self.dataframe['lat'] - self.col_minmax['lat_min']) /
                            ((self.col_minmax['lat_max'] - self.col_minmax['lat_min']) / (self.split - 1)))
        self.dataframe['cell_index'] = y_index * self.split + x_index

        train_val_split, val_test_split = int(num_trips * 0.8), int(num_trips * 0.9)
        self.split_df = [self.dataframe[self.dataframe['trip_id'].isin(select)]
                         for select in (all_trips[:train_val_split],
                                        all_trips[train_val_split:val_test_split],
                                        all_trips[val_test_split:])]

        self.images = {}
        self.trajs = {}
        self.num_channel = 1
        self.num_feat = 1

    def get_images(self, df_index):
        """
        Calculate the spatial-temporal image for trajectories.

        :param df_index:
        :return: the spatial-temporal images with shape (N, C, W, H), the set of OQ-Queries with shape (N, 5),
            and the arrival time labels with shape (N).
        """
        if df_index in self.images:
            images, ODTs, arrive_times = self.images[df_index]
        
        else:
            selected_df = self.split_df[df_index].copy()

            images, ODTs, arrive_times = [], [], []
            for trip_id, group in tqdm(selected_df.groupby('trip_id'),
                                       desc="Gathering images", 
                                       total=selected_df['trip_id'].drop_duplicates().shape[0]):
                ODTs.append((*group.iloc[0][['lng_norm', 'lat_norm']].to_list(),
                             *group.iloc[-1][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[0]['daytime_norm']))  # (5)
                arrive_times.append((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60)
                group['offset'] = (group['time'] - group['time'].iloc[0]) / (group['time'].iloc[-1] - group['time'].iloc[0]) * 2 - 1

                cell_index_diff = group['cell_index'].shift(1) - group['cell_index']
                cell_group = group.loc[cell_index_diff != 0]
                cell_group = cell_group.iloc[::-1]  # This make sure that values on pixels are assigned with values that recorded when the cell is first visited.

                img = np.ones((3, self.split * self.split)) * -1  # (C, W*H)
                array_index = cell_group['cell_index'].astype(int).tolist()
                img[0, array_index] = 1  # Mask
                img[1, array_index] = cell_group['daytime_norm']  # Normalized daytime
                img[2, array_index] = cell_group['offset']  # Normalized time offset
                if not self.flat:
                    img = img.reshape(img.shape[0], self.split, self.split)  # (C, W, H)
                images.append(img)
            
            f_images = []
            if self.flat:
                valid_cells = [image[0][image[0] > 0].sum() for image in images]
                max_valid = int(max(valid_cells))
                for image in tqdm(images, desc='Flatting images'):
                    f_image = np.transpose(image)
                    f_image = f_image[image[0] > 0, :]  # (n_cell, C)
                    f_image = np.concatenate([f_image, np.ones((max_valid - f_image.shape[0], f_image.shape[1])) * -1], 0)  # (max_cell, C)
                    f_images.append(np.transpose(f_image))
            images = f_images

            images, ODTs, arrive_times = np.stack(images, 0), np.array(ODTs), np.array(arrive_times)

            self.images[df_index] = (images, ODTs, arrive_times)
            self.num_channel = images.shape[1]

        full_num_train = images.shape[0]
        partial_index = choices(range(full_num_train), k=math.floor(self.partial * full_num_train)) if df_index == 0 else list(range(full_num_train))
        return (images[partial_index], ODTs[partial_index], arrive_times[partial_index])

    def dump_images(self):
        images = {}
        for i, label in enumerate(['train', 'val', 'test']):
            images |= {f'{label}_image': self.images[i][0],
                       f'{label}_odt': self.images[i][1],
                       f'{label}_arr': self.images[i][2]}
        np.savez(self.image_meta_path, **images)
        print(f'[Dumped meta] to {self.image_meta_path}')

    def load_images(self):
        if os.path.exists(self.image_meta_path):
            loaded_images = np.load(self.image_meta_path)
            for i, label in enumerate(['train', 'val', 'test']):
                self.images[i] = (loaded_images[f'{label}_image'], loaded_images[f'{label}_odt'],
                                  loaded_images[f'{label}_arr'])
            self.num_channel = self.images[0][0].shape[1]
            print(f'[Loaded meta] from {self.image_meta_path}')
        else:
            for i in range(3):
                self.get_images(i)

    def get_trajs(self, df_index):
        """
        Gather the trajectory sequences.

        :return: the trajectories with shape (num_seq, max_len, num_feat), the set of OD-Queries with shape (N, 7), 
            and the arrival time labels with shape (N).
        """
        if df_index in self.trajs:
            trajs, ODTs, arrive_times = self.trajs[df_index]

        else:
            selected_df = self.split_df[df_index].copy()

            trajs, ODTs, arrive_times = [], [], []
            for trip_id, group in tqdm(selected_df.groupby('trip_id'), 
                                       desc='Gathering trajectories', 
                                       total=selected_df['trip_id'].drop_duplicates().shape[0]):
                ODTs.append((group.iloc[0]['cell_index'], *group.iloc[0][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[-1]['cell_index'], *group.iloc[-1][['lng_norm', 'lat_norm']].to_list(),
                             group.iloc[0]['daytime_norm']))  # 7 features
                arrive_times.append((group.iloc[-1]['time'] - group.iloc[0]['time']) / 60)

                group['offset'] = (group['time'] - group['time'].iloc[0]) / (group['time'].iloc[-1] - group['time'].iloc[0]) * 2 - 1
                traj = group[['cell_index', 'lng_norm', 'lat_norm', 'daytime_norm', 'offset']].to_numpy()  # (seq_len, num_feat)
                traj = np.concatenate([traj, np.ones((self.max_len - traj.shape[0], traj.shape[1])) * -1], 0)  # (max_len, num_feat)
                trajs.append(traj)
        
            trajs, ODTs, arrive_times = np.stack(trajs, 0), np.array(ODTs), np.array(arrive_times)
            self.trajs[df_index] = (trajs, ODTs, arrive_times)
            self.num_feat = trajs.shape[1]

        full_num_train = trajs.shape[0]
        partial_index = choices(range(full_num_train), k=math.floor(self.partial * full_num_train)) if df_index == 0 else list(range(full_num_train))
        return (trajs[partial_index], ODTs[partial_index], arrive_times[partial_index])

    def dump_trajs(self):
        meta = {}
        for i, label in enumerate(['train', 'val', 'test']):
            meta |= {f'{label}_traj': self.trajs[i][0],
                     f'{label}_odt': self.trajs[i][1],
                     f'{label}_arr': self.trajs[i][2]}
        np.savez(self.traj_meta_path, **meta)
        print(f'[Dumped meta] to {self.traj_meta_path}')

    def load_trajs(self):
        if os.path.exists(self.traj_meta_path):
            loaded_meta = np.load(self.traj_meta_path)
            for i, label in enumerate(['train', 'val', 'test']):
                self.trajs[i] = (loaded_meta[f'{label}_traj'], 
                                 loaded_meta[f'{label}_odt'],
                                 loaded_meta[f'{label}_arr'])
            self.num_feat = self.trajs[0][0].shape[1]
            print(f'[Loaded meta] from {self.traj_meta_path}')
        else:
            for i in range(3):
                self.get_trajs(i)

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--name', help='the name of the dataset', type=str, required=True)
    parser.add_argument('-s', '--split', help='the number of x and y split', type=int, default=20)
    parser.add_argument('-t', '--type', help='the type of meta information', type=str, default='image')
    parser.add_argument('--flat', action='store_true')
    args = parser.parse_args()

    dataset = TrajectoryDataset(args.name, split=args.split, flat=args.flat)

    if args.type == 'image':
        for i in range(3):
            dataset.get_images(i)
        dataset.dump_images()
    elif args.type == 'traj':
        for i in range(3):
            dataset.get_trajs(i)
        dataset.dump_trajs()
    else:
        raise NotImplementedError()
        