import os
import shutil
import pandas as pd

from random import sample


def join_file_name_features(data_files, file_name_parser, header=False, verbose=False):
    def join_one(data_file):
        file_name_features = file_name_parser(data_file)
        data = pd.read_csv(data_file)
        for k, v in file_name_features.items():
            data[k] = [v] * len(data)
        data.to_csv(data_file, header=header, index=False)

    for f in data_files:
        join_one(f)
        if verbose:
            print('%s joining completed.' % f)


def split_data_files(files, prop, target_dir, rest_dir,
                     clear_if_nonempty=True, reserve_old=True):
    if not os.path.exists(target_dir):
        raise NotADirectoryError('%s is not a valid directory' % target_dir)
    if not os.path.exists(rest_dir):
        raise NotADirectoryError('%s is not a valid directory' % rest_dir)

    target_files = sample(files, int(len(files) * prop))
    rest_files = []
    for file in files:
        if file not in target_files:
            rest_files.append(file)
    move = shutil.copy if reserve_old else shutil.move

    def move_files(src_files, trg_dir):
        if clear_if_nonempty:
            shutil.rmtree(trg_dir)
            os.mkdir(trg_dir)
        for src_file in src_files:
            src_name = os.path.split(src_file)[-1]
            trg_path = os.path.join(trg_dir, src_name)
            move(src_file, trg_path)

    move_files(target_files, target_dir)
    move_files(rest_files, rest_dir)


def split_train_eval_test(files, prop_train, prop_eval, prop_test,
                          train_dir, eval_dir, test_dir, temp_dir=None,
                          clear_if_nonempty=True, reserve_old=True):
    if prop_train < 0 or prop_eval < 0 or prop_test < 0 or not 1. - 1.e-6 < (
            prop_train + prop_eval + prop_test) < 1. + 1.e-6:
        raise ValueError(
            'invalid proportion (prop_train, prop_eval, prop_test), got %f, %f, %f, respectively' % (
                prop_train, prop_eval, prop_test))

    if temp_dir is None:
        temp_dir = 'temp'
        os.mkdir(temp_dir)
    elif not os.path.exists(temp_dir):
        raise NotADirectoryError('invalid temp dir: %s' % temp_dir)

    split_data_files(files, prop_train, train_dir, temp_dir,
                     clear_if_nonempty=clear_if_nonempty, reserve_old=reserve_old)
    eval_test_files = list(map(lambda p: os.path.join(temp_dir, os.path.split(p)[-1]), files))
    split_data_files(eval_test_files, prop_eval, eval_dir, test_dir,
                     clear_if_nonempty=clear_if_nonempty, reserve_old=False)
    if temp_dir == 'temp':
        shutil.rmtree(temp_dir)


def check_file_len(data_files, file_len, title=True):
    invalids = []
    for file in data_files:
        with open(file, 'r') as f:
            if len(f.readlines()) != file_len + int(title):
                invalids.append(file)
    return invalids


# -*- coding: UTF-8 -*-

import os
import shutil
from random import sample

import numpy as np

from data_generator.ctrl_emulator import CtrlEmulator

PARAMS_TO_RECORD = ['xt', 'yt', 'vt', 'theta_t', 'at_normal']


def generate_data():
    em_sim = CtrlEmulator()
    for vdm in range(1, 4):
        for iha in np.arange(-80, -10, 10, dtype=np.float):
            for dm in np.arange(5000, 30000, 100, dtype=np.float):
                name = '{vdm}_{iha}_{dm}.csv'.format(
                    vdm=vdm, iha=iha, dm=dm)
                print(name)
                em_sim.run(initial_params={
                    'value_direction_maneuver': vdm,
                    'initial_heading_angle': iha,
                    'distance_maneuver': dm,
                    'value_target_acceleration': 9.0,
                    't_inter': 0.005},
                    steps=3000,
                    params_to_record=['xt', 'yt', 'vt', 'theta_t', 'at_normal'],
                    data_file='data/{name}'.format(name=name))


def split_train_test(files, prop_train, train_dir=None, test_dir=None):
    train_files = sample(files, int(len(files) * prop_train))
    test_files = []
    for file in files:
        if file not in train_files:
            test_files.append(file)

    if train_dir is None:
        train_dir = 'data/train'
    if test_dir is None:
        test_dir = 'data/test'
    for file in train_files:
        shutil.move(file, os.path.join(train_dir, os.path.split(file)[-1]))
    for file in test_files:
        shutil.move(file, os.path.join(test_dir, os.path.split(file)[-1]))


def main():
    # gen erate_data()
    os.chdir('../')
    data_path = 'data1'
    for _, _, files in os.walk(data_path):
        for i in range(len(files)):
            files[i] = os.path.join(data_path, files[i])
        split_train_test(
            files,
            prop_train=0.8,
            train_dir=os.path.join(data_path, 'train'),
            test_dir=os.path.join(data_path, 'test'))
        break


if __name__ == "__main__":
    em_sim = CtrlEmulator()
    em_sim.run(initial_params={
        'value_direction_maneuver': 2,
        'initial_heading_angle': -50.,
        'distance_maneuver': 15000,
        'value_target_acceleration': 9.0,
        't_inter': 0.005},
        steps=3000,
        params_to_record=['xt', 'yt', 'vt', 'theta_t', 'at_normal'],
        data_file='../data/{name}'.format(name='1.csv'))
