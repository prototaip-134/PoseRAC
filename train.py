import pandas as pd
import numpy as np
import os
import csv
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
from model import PoseRAC
import argparse
import time
import yaml
torch.multiprocessing.set_sharing_strategy('file_system')

def apply_random_flip(landmarks, flip_prob=0.5):
    if np.random.rand() < flip_prob:
        landmarks[:,:,0] = 1 - landmarks[:,:,0]  # Flip x-coordinates
    return landmarks

def apply_random_rotation(landmarks, max_angle=15):
    # Convert max_angle to radians
    max_angle_rad = np.deg2rad(max_angle)
    
    # Generate a random angle within the range of [-max_angle, max_angle]
    angle = np.random.uniform(-max_angle_rad, max_angle_rad)
    
    # Rotation matrix for 2D rotation
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Apply rotation to each landmark
    # Note: Ensure landmarks are in the correct shape: (num_samples, num_landmarks, num_dimensions)
    for i in range(len(landmarks)):
        landmarks[i, :, :2] = landmarks[i, :, :2] @ rotation_matrix.T  # Apply rotation only to x and y coordinates

    return landmarks

def apply_jitter(landmarks, jitter_std=0.01):
    jitter = np.random.normal(0, jitter_std, landmarks.shape)
    landmarks += jitter
    return landmarks


# Original Normalization to improve training robustness.
def normalize_landmarks(all_landmarks, n_landmarks, n_dimensions, training):
    x_max = np.expand_dims(np.max(all_landmarks[:,:,0], axis=1), 1)
    x_min = np.expand_dims(np.min(all_landmarks[:,:,0], axis=1), 1)

    if n_dimensions >= 2:
        y_max = np.expand_dims(np.max(all_landmarks[:,:,1], axis=1), 1)
        y_min = np.expand_dims(np.min(all_landmarks[:,:,1], axis=1), 1)

    if n_dimensions >= 3:
        z_max = np.expand_dims(np.max(all_landmarks[:,:,2], axis=1), 1)
        z_min = np.expand_dims(np.min(all_landmarks[:,:,2], axis=1), 1)

    all_landmarks[:,:,0] = (all_landmarks[:,:,0] - x_min) / (x_max - x_min)

    if n_dimensions >= 2:
        all_landmarks[:,:,1] = (all_landmarks[:,:,1] - y_min) / (y_max - y_min)

    if n_dimensions >= 3:
        all_landmarks[:,:,2] = (all_landmarks[:,:,2] - z_min) / (z_max - z_min)

    # if training:
    #     # Apply flip augmentation
    #     all_landmarks = apply_random_flip(all_landmarks)

    #     # Apply rotation augmentation
    #     all_landmarks = apply_random_rotation(all_landmarks)

    #     # Apply rotation augmentation
    #     all_landmarks = apply_jitter(all_landmarks)

    all_landmarks = all_landmarks.reshape(len(all_landmarks), n_landmarks*n_dimensions)
    return all_landmarks


# Original PoseRAC: For each pose, we use 33 key points to represent it, and each key point has 3 dimensions.
# Here we obtain the pose information (33*3=99) of each key frame, and set up the label (1 for salient pose I and 0 for salient pose II).
def obtain_landmark_label(csv_path, all_landmarks, all_labels, label2index, num_classes, n_landmarks, n_dimensions):
    file_separator=','
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=file_separator)
        for row in csv_reader:
            assert len(row) == n_landmarks * n_dimensions + 2, f'Wrong number of values: Expect: {n_landmarks * n_dimensions + 2}, Actual: {len(row)}'
            landmarks = np.array(row[2:], np.float32).reshape([n_landmarks, n_dimensions])
            all_landmarks.append(landmarks)
            label = label2index[row[1]]

            start_str = row[0].split('/')[-3]
            label_np = np.zeros(num_classes)
            if start_str == 'salient1':
                label_np[label] = 1
            all_labels.append(label_np)
    return all_landmarks, all_labels


def csv2data(train_csv, action2index, num_classes, n_landmarks, n_dimensions, training=True):
    train_landmarks = []
    train_labels = []
    train_landmarks, train_labels = obtain_landmark_label(train_csv, train_landmarks, train_labels, action2index, num_classes, n_landmarks, n_dimensions)

    train_landmarks = np.array(train_landmarks)
    train_labels = np.array(train_labels)
    train_landmarks = normalize_landmarks(train_landmarks, n_landmarks, n_dimensions, training)

    return train_landmarks, train_labels


def main(args):
    old_time = time.time()
    if os.path.isfile(args.config):
        with open(args.config, "r") as fd:
            config = yaml.load(fd, Loader=yaml.FullLoader)
    else:
        raise ValueError("Config file does not exist.")

    csv_label_path = config['dataset']['csv_label_path']
    root_dir = config['dataset']['dataset_root_dir']

    train_csv = os.path.join(root_dir, 'annotation_pose', 'train.csv')
    # valid_csv = os.path.join(root_dir, 'annotation_pose', 'valid.csv')

    label_pd = pd.read_csv(csv_label_path)
    index_label_dict = {}
    length_label = len(label_pd.index)
    for label_i in range(length_label):
        one_data = label_pd.iloc[label_i]
        action = one_data['action']
        label = one_data['label']
        index_label_dict[label] = action
    num_classes = len(index_label_dict)
    action2index = {v: k for k, v in index_label_dict.items()}
    n_landmarks = config['PoseRAC']['n_landmarks']
    n_dimensions = config['PoseRAC']['n_dimensions']

    train_landmarks, train_labels = csv2data(train_csv, action2index, num_classes, n_landmarks, n_dimensions, training=True)
    valid_landmarks, valid_labels = csv2data(train_csv, action2index, num_classes, n_landmarks, n_dimensions, training=False)


    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min',
    )
    ckpt_callback = ModelCheckpoint(mode="min",
                                    monitor="val_loss",
                                    dirpath='./saved_weights',
                                    filename='{epoch}-{val_loss:.2f}',
                                    every_n_epochs=1)

    model = PoseRAC(train_landmarks, train_labels, valid_landmarks, valid_labels, dim=config['PoseRAC']['dim'],
                    heads=config['PoseRAC']['heads'], enc_layer=config['PoseRAC']['enc_layer'],
                    learning_rate=config['PoseRAC']['learning_rate'], seed=config['PoseRAC']['seed'],
                    num_classes=num_classes, alpha=config['PoseRAC']['alpha'])

    trainer = pl.Trainer(callbacks=[early_stop_callback, ckpt_callback], max_epochs=config['trainer']['max_epochs'],
                         auto_lr_find=config['trainer']['auto_lr_find'], accelerator=config['trainer']['accelerator'],
                         devices=config['trainer']['devices'], strategy='ddp')
    
    trainer.tune(model)
    print('Learning rate:', model.learning_rate)
    trainer.fit(model)

    print(f'best loss: {ckpt_callback.best_model_score.item():.5g}')

    weights = model.state_dict()
    torch.save(weights, config['save_ckpt_path'])

    current_time = time.time()
    print('time: ' + str(current_time - old_time) + 's')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate our PoseRAC')
    parser.add_argument('--config', type=str, metavar='DIR',
                        help='path to a config file')
    args = parser.parse_args()
    main(args)
