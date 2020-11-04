#define BOOST_TEST_MODULE 1D - FFTW - Tests

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <gpfft/detail/cufft_wrapper.hpp>
#include <gpfft/detail/fftw_wrapper.hpp>
#include <vector>

using cd = std::complex<double>;
using gpfft::FFT_type;
namespace math = boost::math;

BOOST_AUTO_TEST_CASE(small_transforms_fftw)
{
    const double pi = acos(-1.0);
    std::vector<cd> out;

    out = gpfft::FFTW3<FFT_type::forward>({3});
    BOOST_CHECK(out.size() == 1);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    out = gpfft::FFTW3<FFT_type::backward>({3});
    BOOST_CHECK(out.size() == 1);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    out = gpfft::FFTW3<FFT_type::forward>({1, 1});
    BOOST_CHECK(out.size() == 2);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{2, 0}), 1e-12);
    BOOST_CHECK_SMALL(std::abs(out[1] - cd{0, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    {
        out = gpfft::FFTW3<FFT_type::forward>({1, 1, 1});
        cd w{cos(2 * pi / 3), sin(2 * pi / 3)}, w2 = w * w;
        BOOST_CHECK(out.size() == 3);
        BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
        BOOST_CHECK_SMALL(std::abs(out[1] - 1. - w - w2), 1e-12);
        BOOST_CHECK_SMALL(std::abs(out[2] - 1. - w2 - w2 * w2), 1e-12);
        for (auto c : out)
            BOOST_CHECK(math::isnan(abs(c)) == false);
    }
}
BOOST_AUTO_TEST_CASE(small_transforms_cufft)
{
    const double pi = acos(-1.0);
    std::vector<cd> out;

    out = gpfft::cuFFT<FFT_type::forward>({3});
    BOOST_CHECK(out.size() == 1);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    out = gpfft::cuFFT<FFT_type::backward>({3});
    BOOST_CHECK(out.size() == 1);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    out = gpfft::cuFFT<FFT_type::forward>({1, 1});
    BOOST_CHECK(out.size() == 2);
    BOOST_CHECK_SMALL(std::abs(out[0] - cd{2, 0}), 1e-12);
    BOOST_CHECK_SMALL(std::abs(out[1] - cd{0, 0}), 1e-12);
    for (auto c : out)
        BOOST_CHECK(math::isnan(abs(c)) == false);

    {
        out = gpfft::cuFFT<FFT_type::forward>({1, 1, 1});
        cd w{cos(2 * pi / 3), sin(2 * pi / 3)}, w2 = w * w;
        BOOST_CHECK(out.size() == 3);
        BOOST_CHECK_SMALL(std::abs(out[0] - cd{3, 0}), 1e-12);
        BOOST_CHECK_SMALL(std::abs(out[1] - 1. - w - w2), 1e-12);
        BOOST_CHECK_SMALL(std::abs(out[2] - 1. - w2 - w2 * w2), 1e-12);
        for (auto c : out)
            BOOST_CHECK(math::isnan(abs(c)) == false);
    }
}
