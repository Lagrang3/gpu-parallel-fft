boost_ut = dependency('boost',modules: ['unit_test_framework'],required: false)


test_src += [files(
    '1d-fftw.cu','transpose.cpp','sum.cpp','3d-fftw.cu')]

if ( boost.found() and fftw3.found() and boost_ut.found())
    oned_fftw = executable('1d-fftw',['1d-fftw.cu'],
        include_directories: include,
        dependencies: [boost,fftw3,boost_mpi,boost_ut,cuda])
    test('1d-fftw',oned_fftw)
    
    threed_fftw = executable('3d-fftw',['3d-fftw.cu'],
        include_directories: include,
        dependencies: [boost,fftw3,boost_mpi,mpi,boost_ut,cuda])
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
