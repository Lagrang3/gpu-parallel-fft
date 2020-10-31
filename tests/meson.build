boost_ut = dependency('boost',modules: ['unit_test_framework'],required: false)
fftw3 = dependency('fftw3',required: false)

test_src += [files(
    '1d-fftw.cpp')]

if ( boost_ut.found() and fftw3.found() )
    oned_fftw = executable('1d-fftw',['1d-fftw.cpp'],
        include_directories: include,
        dependencies: [boost_ut,fftw3])
    test('1d-fftw',oned_fftw)
endif
