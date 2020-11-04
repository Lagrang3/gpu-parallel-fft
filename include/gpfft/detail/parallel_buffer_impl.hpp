#pragma once

#include <array>
#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <gpfft/detail/cufft_wrapper.hpp>
#include <gpfft/detail/fftw_wrapper.hpp>
#include <gpfft/fft_type.hpp>
#include <memory>
#include <valarray>

#define for_xyz(i, j, k, nloc)                                                 \
    for (size_t i = 0; i < nloc[0]; ++i)                                       \
        for (size_t j = 0; j < nloc[1]; ++j)                                   \
            for (size_t k = 0; k < nloc[2]; ++k)

namespace gpfft
{
    template <class T>
    void parallel_buff_3D<T>::transpose_reorder()
    {
        size_t nproc = com.size();

        std::valarray<T> tmp(get_local_size());
        std::array<size_t, 3> nloc{N_loc[0] / nproc, N_loc[1],
                                   N_loc[2] * nproc};

        N_loc[0] /= nproc;
        size_t offset = N_loc[0] * N_loc[1] * N_loc[2];
        for (size_t p = 0; p < nproc; ++p)
            for_xyz(x, y, z, N_loc)
            {
                size_t i = x, j = y, k = z + p * N_loc[2];
                tmp[index(i, j, k, nloc)] =
                    (*this)[index(x, y, z) + offset * p];
            }

        N_loc = nloc;
        (*this) = std::move(tmp);
    }

    template <class T>
    void parallel_buff_3D<T>::compute_start_loc()
    {
        int r(N[0] % com.size()), q(N[0] / com.size());
        N_loc[0] = q + (com.rank() < r);
        N_loc[1] = N[1];
        N_loc[2] = N[2];
        start_loc[0] = com.rank() * q + std::min(com.rank(), r);
        start_loc[1] = start_loc[2] = 0;
    }

    template <class T>
    void parallel_buff_3D<T>::transpose_yz()
    {
        std::valarray<T> tmp(get_local_size());
        std::array<size_t, 3> nloc{N_loc[0], N_loc[2], N_loc[1]};

        for_xyz(i, j, k, N_loc) tmp[index(i, k, j, nloc)] =
            (*this)[index(i, j, k)];

        (*this) = std::move(tmp);
        std::swap(N[1], N[2]);
        compute_start_loc();
    }

    template <class T>
    void parallel_buff_3D<T>::transpose_xz()
    {
        std::valarray<T> tmp(get_local_size());
        std::array<size_t, 3> nloc{N_loc[2], N_loc[1], N_loc[0]};

        for_xyz(i, j, k, N_loc) tmp[index(k, j, i, nloc)] =
            (*this)[index(i, j, k)];

        N_loc = nloc;
        (*this) = std::move(tmp);

        all_to_all();
        transpose_reorder();
        std::swap(N[0], N[2]);
        compute_start_loc();
    }

    template <class T>
    void parallel_buff_3D<T>::transpose_xy()
    {
        transpose_yz();
        transpose_xz();
        transpose_yz();
    }

    template <class T>
    T parallel_buff_3D<T>::sum() const
    {
        T s_loc = std::valarray<T>::sum(), s_tot{};
        boost::mpi::all_reduce(com, s_loc, s_tot, std::plus<T>());
        return s_tot;
    }

    template <class T>
    template <FFT_type fft>
    void parallel_buff_3D<T>::local_FFT()
    {
        // with fftw 1-dim fft
        // for (size_t i = 0; i < N_loc[0]; ++i)
        //     for (size_t j = 0; j < N_loc[1]; ++j)
        //         FFTW3<fft>(&(*this)(i, j, 0), &(*this)(i, j + 1, 0),
        //                    &(*this)(i, j, 0));

        // with cufft
        cuFFT<fft>(&((*this)[0]), &((*this)[0]) + this->size(), &((*this)[0]),
                   N_loc[2], N_loc[0] * N_loc[1]);
    }

    template <class T>
    template <FFT_type fft>
    void parallel_buff_3D<T>::local_2dFFT()
    {
        cuFFT2d<fft>(&((*this)[0]), &((*this)[0]) + this->size(), &((*this)[0]),
                     N_loc[2], N_loc[0]);
    }

    // template <class T>
    // template <FFT_type fft>
    // void parallel_buff_3D<T>::FFT3D()
    // {
    //     // FFT on z
    //     local_FFT<fft>();

    //     // FFT on y
    //     transpose_yz();
    //     local_FFT<fft>();
    //     transpose_yz();

    //     // FFT on x
    //     transpose_xz();
    //     local_FFT<fft>();
    //     transpose_xz();
    // }

    template <class T>
    template <FFT_type fft>
    void parallel_buff_3D<T>::FFT3D()
    {
        // FFT on z and y
        local_2dFFT<fft>();

        // FFT on x
        transpose_xz();
        local_FFT<fft>();
        transpose_xz();
    }
    template <class T>
    void parallel_buff_3D<T>::all_to_all()
    {
        const int Ntot = get_local_size();
        const int N = com.size();
        const int Nloc = Ntot / N;

        std::vector<std::vector<T>> sendbuf(N), recvbuf(N);

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < Nloc; ++j)
                sendbuf[i].push_back((*this)[i * Nloc + j]);
        }

        boost::mpi::all_to_all(com, sendbuf, recvbuf);

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < Nloc; ++j)
                (*this)[i * Nloc + j] = recvbuf[i][j];
        }
        com.barrier();
    }

}  // namespace gpfft
