numpy_dep = dependency('numpy')
omp = dependency('openmp', required: false)

deps = [numpy_dep]
if omp.found()
    deps += omp
endif

import('python').find_installation().extension_module(
    '_fusion',
    '_fusion.pyx',
    dependencies: deps,
    subdir: 'geowombat/radiometry',
    install: true
)
