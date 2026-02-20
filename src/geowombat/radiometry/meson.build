py = import('python').find_installation(pure: false)

# Get NumPy include directory
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true
).stdout().strip()

numpy_dep = declare_dependency(include_directories: include_directories(incdir_numpy))
omp = dependency('openmp', required: false)

deps = [numpy_dep]
if omp.found()
    deps += omp
endif

py.extension_module(
    '_fusion',
    '_fusion.pyx',
    dependencies: deps,
    subdir: 'geowombat/radiometry',
    install: true
)
