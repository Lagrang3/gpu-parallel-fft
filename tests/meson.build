boost_ut = dependency('boost',modules: ['unit_test_framework'],required: false)
boost = dependency('boost',required: false)
fftw3 = dependency('fftw3',required: false)
mpirun = find_program('mpirun')


test_src += [files(
    '1d-fftw.cpp','transpose.cpp','sum.cpp','3d-fftw.cpp')]

if ( boost.found() and fftw3.found() and boost_ut.found())
    oned_fftw = executable('1d-fftw',['1d-fftw.cpp'],
        include_directories: include,
        dependencies: [boost,fftw3,boost_mpi,boost_ut])
    test('1d-fftw',oned_fftw)
    
    threed_fftw = executable('3d-fftw',['3d-fftw.cpp'],
        include_directories: include,
        dependencies: [boost,fftw3,boost_mpi,mpi,boost_ut])
    test('3D FFT',mpirun,
       args: ['-np','2',threed_fftw.full_path()],
       is_parallel: false)
    
   transpose = executable('transpose',['transpose.cpp'],
       include_directories: include,
       dependencies: [boost,mpi,boost_mpi,boost_ut])
   test('transpose',mpirun,
       args: ['-np','2',transpose.full_path()],
       is_parallel: false)
   
   sum = executable('sum',['sum.cpp'],
       include_directories: include,
       dependencies: [mpi,boost,boost_mpi,boost_ut])
   test('Parallel Sum',mpirun,
       args: ['-np','2',sum.full_path()],
       is_parallel: false)
endif
    
