test_src += [files('bench.cu')]

bench = executable('bench',['bench.cu'],
    include_directories: include,
    dependencies: [boost,boost_mpi,mpi,cuda,fftw3])
