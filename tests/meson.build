boost_ut = dependency('boost',modules: ['unit_test_framework'],required: false)
fftw3 = dependency('fftw3',required: false)
mpirun = find_program('mpirun')


test_src += [files(
    '1d-fftw.cpp','transpose.cpp','sum.cpp','3d-fftw.cpp')]

if ( boost_ut.found() and fftw3.found() )
    oned_fftw = executable('1d-fftw',['1d-fftw.cpp'],
        include_directories: include,
        dependencies: [boost_ut,fftw3])
    test('1d-fftw',oned_fftw)
    
    threed_fftw = executable('3d-fftw',['3d-fftw.cpp'],
        include_directories: include,
        dependencies: [boost_ut,fftw3,boost_mpi,mpi])
    test('3D FFT',mpirun,
       args: ['-np','2',threed_fftw.full_path()],
       is_parallel: false)
endif

if (boost_ut.found())
   transpose = executable('transpose',['transpose.cpp'],
       include_directories: include,
       dependencies: [boost_ut,mpi,boost_mpi])
   test('transpose',mpirun,
       args: ['-np','2',transpose.full_path()],
       is_parallel: false)
   
   sum = executable('sum',['sum.cpp'],
       include_directories: include,
       dependencies: [boost_ut,mpi,boost_mpi])
   test('Parallel Sum',mpirun,
       args: ['-np','2',sum.full_path()],
       is_parallel: false)
endif
