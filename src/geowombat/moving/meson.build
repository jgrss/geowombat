numpy_dep = dependency('numpy', version: '>=2.0.0',required: true)
import('python').find_installation().extension_module(
  '_moving',
  '_moving.pyx',
  dependencies : [numpy_dep],
  subdir: 'rastercrf/transforms',
  install : true
)
