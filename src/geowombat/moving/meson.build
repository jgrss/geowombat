numpy_dep = dependency('numpy', version: '>=2.0.0',required: true)
omp = dependency('openmp')

import('python').find_installation().extension_module(
  '_moving',
  '_moving.pyx',
  dependencies : [numpy_dep, omp],
  subdir: 'geowombat/transforms',
  install : true
)

