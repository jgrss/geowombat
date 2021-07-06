.. _gpu:

Time series processes on the GPU
================================

Local GPU installation
----------------------

Follow `TensorFlow instructions <https://www.tensorflow.org/install/gpu>`_ or the lines below.

Install the NVIDIA driver::

    sudo apt-get purge nvidia*
    sudo add-apt-repository ppa:graphics-drivers/ppa
    sudo apt-get update
    ubuntu-drivers devices
    sudo apt install nvidia-driver-460

.. note::

    You may have to reboot and disable Secure boot (usually involves holding a key such as F2 on startup)

Download CUDA Toolkit .run file (CUDA 11.1)::

    wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run

Install CUDA Toolkit (CUDA 11.1)::

    sudo sh cuda_11.1.0_455.23.05_linux.run --toolkit --silent --override

Update .profile::

    CUDA_VERSION="11.1"
    export PATH=/usr/local/cuda-${CUDA_VERSION}:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export CUDA_HOME=/usr/local/cuda

Install Python libraries
------------------------

Install JAX (with CUDA 11.1 below)::

    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

Install PyTorch (with CUDA 11.1 below)::

    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

Install Tensorflow (latest versions have GPU support)::

    pip install tensorflow

Basic example
-------------

In the example below, the mean over time is calculated on band 1.

.. code:: python

    import geowombat as gw
    import jax.numpy as jnp

    filenames = ['image1.tif', 'image2.tif', ...]

    with gw.series(filenames) as src:

        src.apply('mean',
                  'temporal_mean.tif',
                  bands=1,
                  num_workers=4)

Stacking multiple statistics
----------------------------

.. code:: python

    with gw.series(filenames) as src:

        src.apply(['mean', 'max', 'cv'],
                  'temporal_stats.tif',
                  bands=1,
                  num_workers=4)

Custom modules
--------------

Custom time series modules can be generated with classes following the format below.

.. code:: python

    class TemporalMean(gw.TimeModule):

        def __init__(self):
            super(TemporalMean, self).__init__()

        def calculate(self, array):

            """
            Args:
                array (``numpy.ndarray`` |
                       ``jax.numpy.DeviceArray`` |
                       ``torch.Tensor`` |
                       ``tensorflow.Tensor``): The input array, shaped [time x bands x rows x columns].

            Returns:
                ``numpy.ndarray`` |
                ``jax.numpy.DeviceArray`` |
                ``torch.Tensor`` |
                ``tensorflow.Tensor``
            """

            # Reduce the time axis, which is the first index position.
            # The output is then shaped [1 x bands x rows x columns] ...
            # so we squeeze the dimensions ...
            # resulting in a returned array of [bands x rows x columns].
            return jnp.nanmean(array, axis=0).squeeze()

.. note::

    ``super(TemporalMean, self).__init__()`` instantiates the base time series module. The only required method
    is :func:`calculate`, which takes one argument. The returned value must be an array shaped
    ``[bands x rows x columns]`` or ``[rows x columns]``.

.. note::

    If ``gw.series(..., transfer_lib='jax')`` then ``jax.numpy`` ``nan`` reductions (e.g., ``jnp.nanmean``) should
    be used because the array data are masked.

To use this class, call it in ``apply``:

.. code:: python

    with gw.series(filenames) as src:

        # Read band 1 and apply the temporal mean reduction
        src.apply(TemporalMean(),
                  'temporal_mean.tif',
                  bands=1,
                  num_workers=4)

Minor changes are needed for multiple band outputs.

First, we add a ``count`` attribute that overrides the default of 1.

.. code:: python

    class TemporalMean(gw.TimeModule):

        def __init__(self):

            super(TemporalMean, self).__init__()

            self.count = 2

        def calculate(self, array):
            return jnp.nanmean(array, axis=0).squeeze()

Then, all is needed is to read the desired bands.

.. code:: python

    with gw.series(filenames) as src:

        # Read bands 1 and 2 and apply the temporal mean reduction
        src.apply(TemporalMean(),
                  'temporal_mean.tif',
                  bands=[1, 2],
                  num_workers=4)

Combining custom modules
------------------------

Combing custom modules is simple. Below, we've created two modules, one to compute the temporal mean and
the other to compute the temporal max. We could use these separately as illustrated above, where both
outputs would generate images with two bands. However, we can also combine the two modules to generate
one 4-band image.

.. ipython:: python

    import geowombat as gw
    import jax.numpy as jnp

    class TemporalMean(gw.TimeModule):
        def __init__(self):
            super(TemporalMean, self).__init__()
            self.count = 2
        def calculate(self, array):
            return jnp.nanmean(array, axis=0).squeeze()

.. ipython:: python

    class TemporalMax(gw.TimeModule):
        def __init__(self):
            super(TemporalMax, self).__init__()
            self.count = 2
        def calculate(self, array):
            return jnp.nanmax(array, axis=0).squeeze()

Combine the two modules

.. code:: python

    stacked_module = gw.TimeModulePipeline([TemporalMean(),
                                            TemporalMax()])

    with gw.series(filenames) as src:

        src.apply(stacked_module,
                  'temporal_stack.tif',
                  bands=[1, 2],
                  num_workers=8)

.. note::

    Modules can also be combined with the ``+`` sign.

For example,

.. ipython:: python

    stacked_module = TemporalMean() + TemporalMax()
    for module in stacked_module.modules:
        print(module)

is equivalent to

.. ipython:: python

    stacked_module = gw.TimeModulePipeline([TemporalMean(),
                                            TemporalMax()])

    for module in stacked_module.modules:
        print(module)

Using the band dictionary
-------------------------

The band dictionary attribute is available within a module if ``band_list`` is provided in the ``apply`` function.

.. code:: python

    class TemporalNDVI(gw.TimeModule):

        def __init__(self):

            super(TemporalNDVI, self).__init__()

            self.count = 1
            self.dtype = 'uint16'

        def calculate(self, array):

            # Set slice tuples for [time, bands, rows, columns]
            sl1 = (slice(0, None), slice(self.band_dict['nir'], self.band_dict['nir']+1), slice(0, None), slice(0, None))
            sl2 = (slice(0, None), slice(self.band_dict['red'], self.band_dict['red']+1), slice(0, None), slice(0, None))

            # Calculate the NDVI
            vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)

            # Scale x10000 (truncating values < 0)
            vi = (jnp.nanmean(array, axis=0) * 10000).astype('uint16')

            return vi.squeeze()

.. code:: python

    with gw.series(filenames) as src:

        # Read band 1 and apply the temporal mean reduction
        src.apply(TemporalNDVI(),
                  'temporal_ndvi.tif',
                  band_list=['red', 'nir'],
                  bands=[3, 4],
                  num_workers=4)

Generic vegetation indices with user arguments
----------------------------------------------

.. code:: python

    class GenericVI(gw.TimeModule):

        def __init__(self, b1, b2):

            super(GenericVI, self).__init__()

            self.b1 = b1
            self.b2 = b2

            self.count = 1
            self.dtype = 'float64'
            self.bigtiff = 'YES'

        def calculate(self, array):

            # Set slice tuples for [time, bands, rows, columns]
            sl1 = (slice(0, None), slice(self.band_dict[self.b2], self.band_dict[self.b2]+1), slice(0, None), slice(0, None))
            sl2 = (slice(0, None), slice(self.band_dict[self.b1], self.band_dict[self.b1]+1), slice(0, None), slice(0, None))

            # Calculate the normalized index
            vi = (array[sl1] - array[sl2]) / ((array[sl1] + array[sl2]) + 1e-9)

            return jnp.nanmean(array, axis=0).squeeze()

Now we can create a pipeline with different band ratios.

.. code:: python

    stacked_module = gw.TimeModulePipeline([GenericVI('red', 'nir'),
                                            GenericVI('green', 'red'),
                                            GenericVI('swir2', 'nir')])

    with gw.series(filenames) as src:

        # Read all bands
        src.apply(stacked_module,
                  'temporal_stack.tif',
                  band_list=['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
                  bands=-1,
                  num_workers=4)

Load and apply PyTorch models
-----------------------------

.. code:: python

    import torch
    import torch.nn.functional as F

    class TorchModel(gw.TimeModule):

        def __init__(self, model_file, model):

            super(TorchModel, self).__init__()

            self.model = model

            checkpoint = torch.load(model_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to('cuda:0')

            self.count = 1
            self.dtype = 'uint8'

        def calculate(self, array):

            torch.cuda.empty_cache()

            logits = self.model(array)
            probas = F.softmax(logits, dim=0)
            labels = probas.argmax(dim=0)

            return labels.squeeze().detach().cpu().numpy()

.. code:: python

    with gw.series(filenames) as src:

        # Read all bands
        src.apply(TorchModel('model.cnn', CNN()),
                  'temporal_stack.tif',
                  transfer_lib='pytorch',
                  band_list=['blue', 'green', 'red', 'nir'],
                  bands=[1, 2, 3, 4],
                  num_workers=4)

Load and apply Tensorflow/Keras models
--------------------------------------

.. code:: python

    import tensorflow as tf

    class TensorflowModel(gw.TimeModule):

        def __init__(self, model_file, model):

            super(TensorflowModel, self).__init__()

            self.model = model
            self.model = tf.keras.models.load_model(model_file)

            self.count = 1
            self.dtype = 'uint8'

        def calculate(self, array):

            labels = self.model.predict(array)

            return labels.eval(session=tf.compat.v1.Session())

.. code:: python

    with gw.series(filenames,
                   window_size=(512, 512),
                   padding=(16, 16, 16, 16)) as src:

        # Read all bands
        src.apply(TensorflowModel('model.cnn', CNN()),
                  'temporal_stack.tif',
                  transfer_lib='tensorflow',
                  band_list=['blue', 'green', 'red', 'nir'],
                  bands=[1, 2, 3, 4],
                  num_workers=4)
