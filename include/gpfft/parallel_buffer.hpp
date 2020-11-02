#pragma once

#include <array>
#include <boost/mpi.hpp>
#include <gpfft/fft_type.hpp>
#include <valarray>

#define _index(i, j, k, nloc) k + nloc[2] * (j + nloc[1] * i)

namespace gpfft
{
    template <class T>
    class parallel_buff_3D : public std::valarray<T>
    {
        using communicator = boost::mpi::communicator;
        communicator com;
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
        void compute_start_loc();

       public:
        using std::valarray<T>::operator=;
        using std::valarray<T>::operator*=;

        parallel_buff_3D(const communicator& in_com,
                         const std::array<size_t, 3>& n)
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

        communicator get_comm() const { return com; }
        // MPI_Comm get_raw_comm() const { return MPI_Comm(com); }

        size_t get_local_size() const { return std::valarray<T>::size(); }
        T sum() const;

        void transpose_yz();
        void transpose_xz();
        void transpose_xy();

        template <FFT_type>
        void FFT3D();

        void report(const char* filename);
    };

}  // namespace gpfft

#include <gpfft/detail/parallel_buffer_impl.hpp>
