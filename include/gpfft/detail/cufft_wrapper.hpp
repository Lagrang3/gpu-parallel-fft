#pragma once
/*
    Wrapper to cuFFT library.
*/

#include "gpfft/fft_type.hpp"
#include <complex>
#include <vector>

#include <cuda.h>
#include <cufft.h>
#include <thrust/device_vector.h>

namespace gpfft
{
    int cuFFT_sign(FFT_type T)
    {
        switch (T)
        {
            case FFT_type::forward:
                return CUFFT_FORWARD;
            case FFT_type::backward:
                return CUFFT_INVERSE;
        }
        return -1;
    }

    // enable if *out_beg == std::complex<double>
    template <FFT_type T, class cRAiterator, class RAiterator>
    void cuFFT(cRAiterator in_beg,
               cRAiterator in_end,
               RAiterator out_beg,
               int Nsize,
               int Nbatch)
    {
        const int n = std::distance(in_beg, in_end);
        assert(n == Nsize * Nbatch);
        cufftHandle plan;
        thrust::device_vector<std::complex<double>> D(in_beg, in_end);

        cufftPlan1d(&plan, Nsize, CUFFT_Z2Z, Nbatch);
        cufftExecZ2Z(plan,
                     reinterpret_cast<cufftDoubleComplex*>(
                         thrust::raw_pointer_cast(&D[0])),
                     reinterpret_cast<cufftDoubleComplex*>(
                         thrust::raw_pointer_cast(&D[0])),
                     cuFFT_sign(T));
        thrust::copy(D.begin(), D.end(), out_beg);
        cufftDestroy(plan);
    }

    template <FFT_type T>
    std::vector<std::complex<double>> cuFFT(
        const std::vector<std::complex<double>>& A)
    {
        std::vector<std::complex<double>> B(A.size());
        cuFFT<T>(A.begin(), A.end(), B.begin(), A.size(), 1);
        return B;
    }

}  // namespace gpfft
