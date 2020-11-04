#define BOOST_TEST_MODULE 3D_FFTW_Tests

#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/mpi.hpp>
#include <boost/test/unit_test.hpp>
#include <gpfft/fft_type.hpp>
#include <gpfft/parallel_buffer.hpp>

#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include <fftw3-mpi.h>

using cd = std::complex<double>;
using gpfft::FFT_type;

namespace ut = boost::unit_test;
namespace mpi = boost::mpi;
namespace math = boost::math;

struct fixture
{
    fixture() { fftw_mpi_init(); }

    ~fixture() {}

    static mpi::environment env;
    static mpi::communicator world;
    static std::default_random_engine rng;
};

mpi::environment fixture::env;
mpi::communicator fixture::world;
std::default_random_engine fixture::rng{111};

BOOST_TEST_GLOBAL_FIXTURE(fixture);

BOOST_AUTO_TEST_CASE(gpfft_stability)
{
    BOOST_REQUIRE(fixture::world.size() == 2);
    const int nc = 14;
    const int localn = nc / fixture::world.size();

    BOOST_REQUIRE(nc == localn * fixture::world.size());

    gpfft::parallel_buff_3D<cd> A(fixture::world, {nc, nc, nc});
    std::uniform_real_distribution<double> U(0, 1);

    for (size_t i = 0; i < A.size(); ++i)
        A[i] = U(fixture::rng);

    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.local_FFT<FFT_type::forward>();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.transpose_yz();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.local_FFT<FFT_type::forward>();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.transpose_yz();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.transpose_xz();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.local_FFT<FFT_type::forward>();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);

    A.transpose_xz();
    for (size_t i = 0; i < A.size(); ++i)
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);
}

BOOST_AUTO_TEST_CASE(gpfft_vs_fftw3_parallel)
{
    BOOST_REQUIRE(fixture::world.size() == 2);
    const int nc = 14;
    gpfft::parallel_buff_3D<cd> A(fixture::world, {nc, nc, nc});
    std::uniform_real_distribution<double> U(0, 1);

    for (size_t i = 0; i < A.size(); ++i)
        A[i] = U(fixture::rng);

    // fftw3-3d

    ptrdiff_t local_n, local_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(nc, nc, nc, fixture::world,
                                                   &local_n, &local_start);
    fftw_complex* data = fftw_alloc_complex(alloc_local);

    fftw_plan p = fftw_mpi_plan_dft_3d(nc, nc, nc, data, data, fixture::world,
                                       FFTW_FORWARD, FFTW_ESTIMATE);

    const size_t local_size = local_n * nc * nc;
    BOOST_REQUIRE(local_size == A.size());

    for (size_t i = 0; i < local_size; ++i)
    {
        data[i][0] = A[i].real(), data[i][1] = A[i].imag();
    }

    fftw_execute(p);

    A.FFT3D<gpfft::FFT_type::forward>();
    for (size_t i = 0; i < A.size(); ++i)
    {
        BOOST_REQUIRE(math::isnan(std::abs(A[i])) == false);
    }

    double diff = 0;
    for (size_t i = 0; i < local_size; ++i)
    {
        diff += std::abs(A[i] - cd(data[i][0], data[i][1]));
    }

    mpi::all_reduce(fixture::world, diff, std::plus<double>());

    fftw_free(data);
    fftw_destroy_plan(p);

    BOOST_CHECK(math::isnan(diff) == false);
    BOOST_CHECK_SMALL(diff, 1e-11);
}
