project('gpfft','cpp','cuda',
        default_options : ['cpp_std=c++17',
        'warning_level=3'],
        version: '0.0.1')
add_global_arguments(['-Ofast','-march=native' ],language : 'cpp')
add_global_arguments(['--disable-warnings' ],language : 'cuda')

cpp=meson.get_compiler('cpp')
nvcc=meson.get_compiler('cuda')
mpi=dependency('mpi',language: 'cpp')
boost_mpi = dependency('boost',modules: ['mpi','serialization'])
cuda = dependency('cuda',modules: ['cufft'])

headers=[]
sources=[]
test_src=[]

subdir('include')
#subdir('sources')
subdir('tests')

_format = find_program('clang-format', required:false)

if (_format.found())
  run_target('format', command: [_format, '-i', headers,test_src,sources])
endif
