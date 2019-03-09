from mpglue import moving_window
import numpy as np


class MovingMethods(object):

    def __init__(self, array, window_size):

        self._array = array
        self._window_size = window_size

    def min(self):

        if len(self._array.shape) > 2:

            self._array = np.float32(self._array)

            for lidx, layer in enumerate(self._array):

                self._array[lidx] = moving_window(layer,
                                                  statistic='min',
                                                  window_size=self._window_size)

        else:

            self._array[:] = moving_window(self._array,
                                           statistic='min',
                                           window_size=self._window_size)

        return self._array

    def max(self):

        if len(self._array.shape) > 2:

            self._array = np.float32(self._array)

            for lidx, layer in enumerate(self._array):

                self._array[lidx] = moving_window(layer,
                                                  statistic='max',
                                                  window_size=self._window_size)

        else:

            self._array[:] = moving_window(self._array,
                                           statistic='max',
                                           window_size=self._window_size)

        return self._array

    def mean(self):

        if len(self._array.shape) > 2:

            self._array = np.float32(self._array)

            for lidx, layer in enumerate(self._array):

                self._array[lidx] = moving_window(layer,
                                                  statistic='mean',
                                                  window_size=self._window_size)

        else:

            self._array[:] = moving_window(self._array,
                                           statistic='mean',
                                           window_size=self._window_size)

        return self._array

    def std(self):

        if len(self._array.shape) > 2:

            self._array = np.float32(self._array)

            for lidx, layer in enumerate(self._array):

                self._array[lidx] = moving_window(layer,
                                                  statistic='std',
                                                  window_size=self._window_size)

        else:

            self._array[:] = moving_window(self._array,
                                           statistic='std',
                                           window_size=self._window_size)

        return self._array

    def sum(self):

        if len(self._array.shape) > 2:

            self._array = np.float32(self._array)

            for lidx, layer in enumerate(self._array):

                self._array[lidx] = moving_window(layer,
                                                  statistic='sum',
                                                  window_size=self._window_size)

        else:

            self._array[:] = moving_window(self._array,
                                           statistic='sum',
                                           window_size=self._window_size)

        return self._array


class MovingWindow(object):

    def moving(self, window_size):
        return MovingMethods(self, window_size)
