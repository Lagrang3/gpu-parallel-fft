#pragma once

#include <array>
#include <boost/mpi.hpp>
#include <gpfft/fft_type.hpp>
#include <valarray>

namespace gpfft
{
    template <class T>
    class parallel_buff_3D : public std::valarray<T>
    {
        using communicator = boost::mpi::communicator;
        communicator com;
        std::array<size_t, 3> N, N_loc, start_loc;

        void transpose_reorder();
        void all_to_all();

        void compute_start_loc();

        inline size_t index(size_t i,
                            size_t j,
                            size_t k,
                            std::array<size_t, 3> nloc) const noexcept
        {
            return k + nloc[2] * (j + nloc[1] * i);
        }
        inline size_t index(size_t i, size_t j, size_t k) const noexcept
        {
            return k + N_loc[2] * (j + N_loc[1] * i);
        }

       public:
        using std::valarray<T>::operator=;
        using std::valarray<T>::operator*=;

        template <FFT_type fft>
        void local_FFT();

        parallel_buff_3D(const communicator& in_com,
                         const std::array<size_t, 3>& n)
            : com(in_com), N(n)
        {
            compute_start_loc();
            std::valarray<T>::resize(N_loc[0] * N_loc[1] * N_loc[2]);
        }

        T& operator()(size_t x, size_t y, size_t z)
        {
            return (*this)[index(x, y, z)];
        }
        const T& operator()(size_t x, size_t y, size_t z) const
        {
            return (*this)[index(x, y, z)];
        }

        auto get_N() const { return N; }
        auto get_nloc() const { return N_loc; }
        auto get_ploc() const { return start_loc; }

        communicator get_comm() const { return com; }

        size_t get_local_size() const { return std::valarray<T>::size(); }
        T sum() const;

        void transpose_yz();
        void transpose_xz();
        void transpose_xy();

        template <FFT_type>
        void FFT3D();
    };

}  // namespace gpfft

#include <gpfft/detail/parallel_buffer_impl.hpp>
