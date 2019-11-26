#!/usr/bin/env python

import os
from pathlib import Path
import argparse

from satts.errors import logger
import geowombat as gw
from geowombat.models import CloudClassifier
import satsmooth

import numpy as np
import numba as nb
import pandas as pd
import geopandas as gpd
import sklearn_crfsuite
import joblib
from tqdm import tqdm, trange


SENSOR_BANDS = dict(l7=dict(blue=0,
                            green=1,
                            red=2,
                            nir=3,
                            swir1=4,
                            swir2=5),
                    s2l7=dict(blue=0,
                              green=1,
                              red=2,
                              nir=3,
                              swir1=4,
                              swir2=5))


@nb.jit
def sample_to_dict(tsp, *args):

    """
    Args:
        tsp (1d)
    """

    nt = len(tsp)

    if args:

        n_args = len(args)

        tlab = np.empty(nt+n_args, dtype=object)
        tval = np.zeros(nt+n_args, dtype='float64')

    else:

        tlab = np.empty(nt, dtype=object)
        tval = np.zeros(nt, dtype='float64')

    for r in range(0, nt):

        tlab[r] = str(r + 1).encode('utf-8')
        tval[r] = tsp[r]

    if args:

        for i, (arg_name, arg_value) in enumerate(args):

            tlab[r+i+1] = arg_name.encode('utf-8')
            tval[r+i+1] = arg_value

    return dict(zip(tlab.tolist(), tval))


def _evi2(red, nir):
    """Two-band Enhanced Vegetation Index"""
    return 2.5 * ((nir - red) / (nir + 1.0 + (2.4 * red)))


def _bsi(blue, red, nir, swir2):
    """Bare Soil Index"""
    return ((swir2 + red) - (nir - blue)) / ((swir2 + red) + (nir - blue))


def _brightness_swir(green, red, nir, swir1):
    """Brightness Index"""
    return (green**2 + red**2 + nir**2 + swir1**2)**0.5


def _nbr(nir, swir2):
    """Normalized Burn Ratio"""
    return (nir - swir2) / (nir + swir2)


def _ndmi(nir, swir1):
    """Normalized Difference Moisture Index"""
    return (nir - swir1) / (nir + swir1)


def _ndvi(red, nir):
    """Normalized Difference Vegetation Index"""
    return (nir - red) / (nir + red)


def _wi(red, swir1):
    """Woody Index"""
    return 0.0 if red + swir1 > 0.5 else 1.0 - ((red + swir1) / 0.5)


@nb.jit
def array_to_dict(sensor, *args):

    """
    Converts an array sample to a CRF features
    """

    nargs = len(args)
    feas = list()

    blue_idx = SENSOR_BANDS[sensor]['blue']
    green_idx = SENSOR_BANDS[sensor]['green']
    red_idx = SENSOR_BANDS[sensor]['red']
    nir_idx = SENSOR_BANDS[sensor]['nir']
    swir1_idx = SENSOR_BANDS[sensor]['swir1']
    swir2_idx = SENSOR_BANDS[sensor]['swir2']

    for si in range(0, nargs):

        tsamp = args[si] * 0.0001

        ndvi = _ndvi(tsamp[red_idx], tsamp[nir_idx])
        evi2 = _evi2(tsamp[red_idx], tsamp[nir_idx])
        #bsi = _bsi(tsamp[blue_idx], tsamp[red_idx], tsamp[nir_idx], tsamp[swir2_idx])
        brightness = _brightness_swir(tsamp[green_idx],
                                      tsamp[red_idx],
                                      tsamp[nir_idx],
                                      tsamp[swir1_idx])
        wi = _wi(tsamp[red_idx], tsamp[swir1_idx])
        ndmi = _ndmi(tsamp[nir_idx], tsamp[swir1_idx])
        nbr = _nbr(tsamp[nir_idx], tsamp[swir2_idx])

        feas.append(sample_to_dict(tsamp,
                                   ('brightness', brightness),
                                   ('evi2', evi2),
                                   ('nbr', nbr),
                                   ('ndmi', ndmi),
                                   ('ndvi', ndvi),
                                   ('wi', wi)))

    return feas


def nd_to_columns(array2reshape, layers, rows, columns):

    """
    Reshapes an array from nd layout to [samples (rows*columns) x dimensions]
    """

    if layers == 1:
        return array2reshape.flatten()[:, np.newaxis]
    else:
        return array2reshape.reshape(layers, rows, columns).transpose(1, 2, 0).reshape(rows * columns, layers)


def columns_to_nd(array2reshape, layers, rows, columns):

    """
    Reshapes an array from columns layout to [n layers x rows x columns]
    """

    if layers == 1:
        return array2reshape.reshape(columns, rows).T
    else:
        return array2reshape.T.reshape(layers, rows, columns)


def fill_nodata(X_data, scale_factor=0.0001):

    ntime, nlayers, nrows, ncols = X_data.shape

    for layer in range(0, nlayers):

        X_data_layer = X_data[:, layer, :, :]

        X_data_layer_ = X_data_layer * scale_factor
        X_data_layer_[X_data_layer_ > 1] = 1

        ysm = satsmooth.interp2d(np.ascontiguousarray(nd_to_columns(X_data_layer_,
                                                                    ntime,
                                                                    nrows,
                                                                    ncols), dtype='float32'),
                                 no_data_value=32767*scale_factor)

        X_data[:, layer, :, :] = columns_to_nd(ysm, ntime, nrows, ncols)

    for layer in range(0, nlayers):

        X_data[:, layer, :, :] = satsmooth.spatial_temporal(
            np.ascontiguousarray(X_data[:, layer, :, :], dtype='float32'),
            k=3,
            t=3,
            sigma_time=0.0,
            sigma_color=0.1,
            sigma_space=0.1,
            n_jobs=8)

    return X_data / scale_factor


def prepare_data(df, rpath, exclude=None, image_ext='.img'):

    minrow = 1e9
    mincol = 1e9

    X_data = list()
    y_data = list()

    for row in tqdm(df.itertuples(index=True, name='Pandas'), total=df.shape[0]):

        rimage = rpath.joinpath(row.image + image_ext)

        if rimage.is_file():

            with gw.open(rimage.as_posix(), chunks=1024) as ds:

                if exclude:

                    if row.label in exclude:
                        continue

                clip = gw.clip(ds,
                               df,
                               query="index == {:d}".format(row.Index),
                               mask_data=False)

                subset = clip.data.compute()

                minrow = min(subset.shape[1], minrow)
                mincol = min(subset.shape[2], mincol)

                X_data.append(subset)
                y_data.append(row.label)

    X_data = np.array([d[:, :minrow, :mincol] for d in X_data], dtype='float64')

    X_data = fill_nodata(X_data, scale_factor=0.0001)

    return X_data, y_data


def main():

    rpath = Path('/scratch/rsc8/hardtkel/rapidfires/LW')
    training_data = '/scratch/rsc4/graesser/temp/s2/training/s2_training.shp'
    out_model = '/scratch/rsc4/graesser/temp/clouds.model'

    n_iter = 10
    sensor = 's2l7'
    max_rand_length = 10

    crf_params = dict(algorithm='lbfgs',
                      c1=0.001,
                      c2=0.001,
                      max_iterations=2000,
                      num_memories=20,
                      epsilon=0.01,
                      delta=0.01,
                      period=20,
                      linesearch='MoreThuente',  # 'MoreThuente' 'Backtracking' 'StrongBacktracking'
                      max_linesearch=20,
                      all_possible_states=True,
                      all_possible_transitions=True,
                      verbose=False)

    #############################################

    vpath = Path(training_data)

    # Load the data

    logger.info('  Loading the data ...')

    df = gpd.read_file(vpath.as_posix())

    X_data, y_data = prepare_data(df, rpath)

    # Augment the data

    logger.info('  Data augmentation ...')

    ntime, nbands, nrows, ncols = X_data.shape

    X = list()
    y = list()

    for iter_ in trange(0, n_iter):

        for a in [['u', 'w', 'h'], ['u', 'l', 'h']]:

            Xd = list()
            yd = list()

            idx_null = np.array([ij for ij, cl in enumerate(y_data) if cl not in a], dtype='int64')

            ntime_null = idx_null.shape[0]

            low = np.random.randint(0, high=int(ntime_null / 2))
            high = np.random.randint(low + 2, high=ntime_null)
            idx_random = np.array([i + 1 for i in range(low, high)], dtype='int64')

            n_rand = min(len(idx_random), max_rand_length)

            for i in range(0, n_rand):
                # Get a random subset of temporal indices
                idx = idx_null[np.random.choice(idx_random, size=n_rand, replace=False)]

                # Transpose each temporal state --> samples x features
                Xd_ = [dlayer.transpose(1, 2, 0).reshape(nrows * ncols, nbands) for dlayer in X_data[idx]]

                # Flatten the data from [time x features x rows x columns] --> [s1, s2, ..., sn]
                # len(Xd_) = n samples
                # len(Xd_[0]) = n time
                Xd_ = [array_to_dict(sensor, *[Xd_[j][i] for j in range(0, idx.shape[0])]) for i in
                       range(0, nrows * ncols)]

                # len(y_) = n samples
                # len(y_[0]) = n time
                y_ = [np.array(y_data)[idx].tolist() for i in range(0, nrows * ncols)]

                Xd += Xd_
                yd += y_

            X += Xd
            y += yd

    logger.info('  Number of random arrangements: {:,d}'.format(n_iter))
    logger.info('  Number of samples: {:,d}'.format(nrows * ncols))
    logger.info('  Number of random arrangements x n samples: {:,d}'.format(len(X)))
    logger.info('')
    logger.info('  Fitting the model ...')

    # Gradient descent using the limited-memory BFGS method (with L1 and L2 regularization)
    model = CloudClassifier(**crf_params)

    model.fit(X, y)

    model.to_file(out_model, overwrite=True)

    logger.info('  Saved the model to {}'.format(out_model))


if __name__ == '__main__':
    main()
