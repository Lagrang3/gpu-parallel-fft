#pragma once
/*
    Wrapper to FFTW3 library.
*/

#include "gpfft/fft_type.hpp"
#include <cassert>
#include <complex>
#include <fftw3.h>
#include <vector>

namespace gpfft
{
    int FFTW_sign(FFT_type T)
    {
        switch (T)
        {
            case FFT_type::forward:
                return FFTW_FORWARD;
            case FFT_type::backward:
                return FFTW_BACKWARD;
        }
        return -1;
    }

    // enable if *out_beg == std::complex<double>
    template <FFT_type T, class cRAiterator, class RAiterator>
    void FFTW3(cRAiterator in_beg, cRAiterator in_end, RAiterator out_beg)
    {
        const int n = std::distance(in_beg, in_end);
        assert(n > 0);
        fftw_plan plan;
        fftw_complex* data;

        data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
        plan = fftw_plan_dft_1d(n, data, data, FFTW_sign(T), FFTW_ESTIMATE);

        for (int i = 0; i < n; ++i)
        {
            data[i][0] = in_beg->real();
            data[i][1] = in_beg->imag();

            ++in_beg;
        }
        /*for (fftw_complex* data_beg = data; in_beg != in_end;
             ++in_beg, ++data_beg)
            *data_beg[0] = in_beg->real(), *data_beg[1] = in_beg->imag();*/

        fftw_execute(plan);

        /*
        for (fftw_complex *beg = out, *end = out + n;
             beg != end; ++beg, ++out_beg)
            *out_beg = {*beg[0], *beg[1]};
        */
        for (int i = 0; i < n; ++i)
        {
            *out_beg = std::complex<double>{data[i][0], data[i][1]};
            ++out_beg;
        }

        fftw_free(data);
        fftw_destroy_plan(plan);
    }

    template <FFT_type T>
    std::vector<std::complex<double>> FFTW3(
        const std::vector<std::complex<double>>& A)
    {
        std::vector<std::complex<double>> B(A.size());
        FFTW3<T>(A.begin(), A.end(), B.begin());
        return B;
    }

}  // namespace gpfft
