#define BOOST_TEST_MODULE 1D-FFTW-Tests

#include <boost/test/unit_test.hpp>
#include <gpfft/detail/fftw_wrapper.hpp>
#include <vector>
#include <complex>
#include <cmath>

using cd = std::complex<double>;
using gpfft::FFT_type;

BOOST_AUTO_TEST_CASE(small_transforms)
{
    const double pi = acos(-1.0);
    std::vector<cd> out;
    
    out = gpfft::FFTW3<FFT_type::forward>({3});
    BOOST_CHECK(out.size()==1);
    BOOST_CHECK_SMALL(std::abs(out[0]-cd{3,0}),1e-12);
    
    out = gpfft::FFTW3<FFT_type::backward>({3});
    BOOST_CHECK(out.size()==1);
    BOOST_CHECK_SMALL(std::abs(out[0]-cd{3,0}),1e-12);
    
    out = gpfft::FFTW3<FFT_type::forward>({1,1});
    BOOST_CHECK(out.size()==2);
    BOOST_CHECK_SMALL(std::abs(out[0]-cd{2,0}),1e-12);
    BOOST_CHECK_SMALL(std::abs(out[1]-cd{0,0}),1e-12);
    
    {
    out = gpfft::FFTW3<FFT_type::forward>({1,1,1});
    cd w{ cos(2*pi/3),sin(2*pi/3) },w2=w*w;
    BOOST_CHECK(out.size()==3);
    BOOST_CHECK_SMALL(std::abs(out[0]-cd{3,0}),1e-12);
    BOOST_CHECK_SMALL(std::abs(out[1]-1.-w-w2),1e-12);
    BOOST_CHECK_SMALL(std::abs(out[2]-1.-w2 - w2*w2),1e-12);
    }
}
