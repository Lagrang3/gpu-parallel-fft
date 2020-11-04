#include <boost/mpi.hpp>
#include <gpfft/gpfft.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <cassert>
#include <complex>
#include <random>
#include <vector>
#include <chrono>

#include <fftw3-mpi.h>

using cd = std::complex<double>;
using gpfft::FFT_type;
using microseconds = std::chrono::microseconds;

namespace math = boost::math;
namespace mpi = boost::mpi;

int main()
{
    // initialize
    
    mpi::environment env;
    mpi::communicator world;
    std::default_random_engine rng{111};
    
    const int Nx= 512;
    assert(Nx % world.size() == 0);
    gpfft::parallel_buff_3D<cd> A(world, {Nx,Nx,Nx});
    
    std::uniform_real_distribution<double> U(0, 1);

    for (size_t i = 0; i < A.size(); ++i)
        A[i] = U(rng);
    
    // .. code here for fftw
    ptrdiff_t local_n, local_start;
    ptrdiff_t alloc_local = fftw_mpi_local_size_3d(Nx,Nx,Nx,world,
                                                   &local_n, &local_start);
    fftw_complex* data = fftw_alloc_complex(alloc_local);

    fftw_plan p = fftw_mpi_plan_dft_3d(Nx,Nx,Nx, data, data,world,
                                       FFTW_FORWARD, FFTW_ESTIMATE);

    const size_t local_size = local_n * Nx*Nx;
    assert(local_size == A.size());

    for (size_t i = 0; i < local_size; ++i)
    {
        data[i][0] = A[i].real(), data[i][1] = A[i].imag();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    fftw_execute(p);
    auto dt = std::chrono::high_resolution_clock::now() - t1;
    double musec_fftw = std::chrono::duration_cast<microseconds>(dt).count();
    
    t1 = std::chrono::high_resolution_clock::now();
    A.FFT3D<gpfft::FFT_type::forward>();
    dt = std::chrono::high_resolution_clock::now() - t1;
    double musec_cufft = std::chrono::duration_cast<microseconds>(dt).count();
    
    // compare outputs
    for (size_t i = 0; i < A.size(); ++i)
    {
        assert(math::isnan(std::abs(A[i])) == false);
    }
    double diff = 0;
    for (size_t i = 0; i < local_size; ++i)
    {
        diff += std::abs(A[i] - cd(data[i][0], data[i][1]));
    }
    mpi::all_reduce(world, diff, std::plus<double>());
    assert(math::isnan(diff) == false);
    
    
    // release resources
    fftw_free(data);
    fftw_destroy_plan(p);
    
    if(world.rank()==0)
    std::cout 
        << "Time (fftw): " << musec_fftw*1e-3 << " ms\n"
        << "Time (cufft): " << musec_cufft*1e-3 << " ms\n"
        << "Diff: " << diff << '\n';
    
    return 0;
}
