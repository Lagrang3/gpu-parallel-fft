project('gpfft','cpp',
        default_options : ['cpp_std=c++17',
        'warning_level=3','optimization=3'],
        version: '0.0.1')
add_global_arguments(['-Ofast','-march=native' ],language : 'cpp')

cpp=meson.get_compiler('cpp')
mpi=dependency('mpi',language: 'cpp')
boost_mpi = dependency('boost',modules: ['mpi'])

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
