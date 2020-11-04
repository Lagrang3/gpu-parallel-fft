project('gpfft','cpp','cuda',
        default_options : ['cpp_std=c++17',
        'warning_level=3'],
        version: '0.0.1')
add_global_arguments(['-Ofast','-march=native' ],language : 'cpp')
add_global_arguments(['--disable-warnings','-O3' ],language : 'cuda')

cpp=meson.get_compiler('cpp')
nvcc=meson.get_compiler('cuda')
mpi=dependency('mpi',language: 'cpp')
boost_mpi = dependency('boost',modules: ['mpi','serialization'])
boost = dependency('boost')
cuda = dependency('cuda',modules: ['cufft'])
mpirun = find_program('mpirun')
fftw3 = dependency('fftw3',required: false)

headers=[]
sources=[]
test_src=[]

subdir('include')
subdir('tests')
subdir('benchmarks')

_format = find_program('clang-format', required:false)

if (_format.found())
  run_target('format', command: [_format, '-i', headers,test_src,sources])
endif
