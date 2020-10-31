#pragma once

#include "fft.h"
#include "mpi_handler.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <valarray>

#define for_xyz(i, j, k, nloc)                                                 \
    for (size_t i = 0; i < nloc[0]; ++i)                                       \
        for (size_t j = 0; j < nloc[1]; ++j)                                   \
            for (size_t k = 0; k < nloc[2]; ++k)

#define _index(i, j, k, nloc) k + nloc[2] * (j + nloc[1] * i)

namespace gpfft
{
    template <class T>
    class parallel_buff_3D : public std::valarray<T>
    {
        mpi_comm com;
        std::array<size_t, 3> N;

        void transpose_reorder();
        void all_to_all();
        void get_slice(int writer, T* buf, int pos);
        void write_slice(int writer,
                         T* buf,
                         std::ofstream& ofs,
                         std::string head);

       public:
        std::array<size_t, 3> N_loc, start_loc;

       private:
        void compute_start_loc()
        {
            int r(N[0] % com.size()), q(N[0] / com.size());
            N_loc[0] = q + (com.rank() < r);
            N_loc[1] = N[1];
            N_loc[2] = N[2];
            start_loc[0] = com.rank() * q + std::min(com.rank(), r);
            start_loc[1] = start_loc[2] = 0;
        }

       public:
        using std::valarray<T>::operator=;
        using std::valarray<T>::operator*=;

        parallel_buff_3D(const mpi_comm& in_com, const std::array<size_t, 3>& n)
            : com(in_com), N(n)
        {
            compute_start_loc();
            std::valarray<T>::resize(N_loc[0] * N_loc[1] * N_loc[2]);
        }

        T& operator()(size_t x, size_t y, size_t z)
        {
            return (*this)[_index(x, y, z, N_loc)];
        }
        const T& operator()(size_t x, size_t y, size_t z) const
        {
            return (*this)[_index(x, y, z, N_loc)];
        }

        auto get_N() const { return N; }
        auto get_nloc() const { return N_loc; }
        auto get_ploc() const { return start_loc; }

        MPI_Comm get_com() const { return com.get_com(); }

        mpi_comm get_comm() const { return com; }

        size_t get_local_size() const { return std::valarray<T>::size(); }
        T sum() const;

        void transpose_yz();
        void transpose_xz();
        void transpose_xy();

        /* todo: mpi communication for template T */
        void FFT3D(const std::array<T, 3> e, const T _1 = T(1));

        void report(const char* filename);
    };

}  // namespace gpfft
