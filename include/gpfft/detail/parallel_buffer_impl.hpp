#pragma once

#include <algorithm>
#include <array>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cassert>
#include <fstream>
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
                tmp[_index(i, j, k, nloc)] =
                    (*this)[_index(x, y, z, N_loc) + offset * p];
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

        for_xyz(i, j, k, N_loc) tmp[_index(i, k, j, nloc)] =
            (*this)[_index(i, j, k, N_loc)];

        (*this) = std::move(tmp);
        std::swap(N[1], N[2]);
        compute_start_loc();
    }

    template <class T>
    void parallel_buff_3D<T>::transpose_xz()
    {
        std::valarray<T> tmp(get_local_size());
        std::array<size_t, 3> nloc{N_loc[2], N_loc[1], N_loc[0]};

        for_xyz(i, j, k, N_loc) tmp[_index(k, j, i, nloc)] =
            (*this)[_index(i, j, k, N_loc)];

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
    void parallel_buff_3D<T>::report(const char* filename)
    {
        // using std::cerr;
        int r = 0;
        // cerr << "start Parallel report "<<r<<" "<<com.rank()<<"\n";r++;
        int writer = 0, Ntot;
        std::ofstream ofs;
        std::unique_ptr<T[]> buf;

        if (com.rank() == writer)
            ofs.open(filename);

        // get a slice yz
        // configuration is: xyz

        Ntot = N_loc[1] * N_loc[2];
        buf.reset(new T[Ntot]);
        get_slice(writer, buf.get(), N[0] / 2);
        // cerr << "get slice yz Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++;
        write_slice(writer, buf.get(), ofs, "y z");
        // cerr << "write yz Parallel report "<<r<<" "<<com.rank()<<"\n";r++;

        // get a slice xz
        transpose_xy();
        // cerr << "transposer xy Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++; configuration is: yxz

        Ntot = N_loc[1] * N_loc[2];
        buf.reset(new T[Ntot]);
        get_slice(writer, buf.get(), N[0] / 2);
        // cerr << "get slice xz Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++;
        write_slice(writer, buf.get(), ofs, "x z");
        // cerr << "write xz Parallel report "<<r<<" "<<com.rank()<<"\n";r++;

        // get a slice xy
        transpose_xz();
        // cerr << "transpose xz Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++; configuration is: zxy

        Ntot = N_loc[1] * N_loc[2];
        buf.reset(new T[Ntot]);
        get_slice(writer, buf.get(), N[0] / 2);
        // cerr << "get slice xy Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++;
        write_slice(writer, buf.get(), ofs, "x y");
        // cerr << "write xy Parallel report "<<r<<" "<<com.rank()<<"\n";r++;

        // go back to normal
        transpose_xz();
        // cerr << "transpose xz Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++; configuration is: yxz
        transpose_xy();
        // cerr << "transpoze xz Parallel report "<<r<<"
        // "<<com.rank()<<"\n";r++; configuration is: xyz
    }

    template <class T>
    void parallel_buff_3D<T>::get_slice(int writer, T* buf, int pos)
    {
        int sender = 0, lpos = pos - start_loc[0], Ntot = N_loc[1] * N_loc[2];

        if (lpos >= 0 and lpos < N_loc[0])
            sender = com.rank();

        boost::mpi::all_reduce(com, sender, std::plus<int>());

        if (sender == com.rank())
        {
            for (int i = 0; i < N_loc[1]; ++i)
                for (int j = 0; j < N_loc[2]; ++j)
                    buf[i * N_loc[2] + j] = (*this)(lpos, i, j);
        }

        if (sender != writer)
        {
            if (sender == com.rank())
                MPI_Send(buf, Ntot * sizeof(T), MPI_CHAR, writer, 1, com);
            if (writer == com.rank())
                MPI_Recv(buf, Ntot * sizeof(T), MPI_CHAR, sender, 1, com,
                         MPI_STATUS_IGNORE);
        }
        com.barrier();
    }

    template <class T>
    void parallel_buff_3D<T>::write_slice(int writer,
                                          T* buf,
                                          std::ofstream& ofs,
                                          std::string head)
    {
        if (com.rank() == writer)
        {
            ofs << head << '\n';
            ofs << N_loc[1] << ' ' << N_loc[2] << '\n';
            for (int i = 0; i < N_loc[1]; ++i)
            {
                for (int j = 0; j < N_loc[2]; ++j)
                    ofs << buf[i * N_loc[2] + j] << ' ';
                ofs << '\n';
            }
        }
        com.barrier();
    }

    template <>
    double parallel_buff_3D<double>::sum() const
    {
        double s_loc = 0, s_tot = 0;
        s_loc = std::valarray<double>::sum();
        boost::mpi::all_reduce(com, s_loc, s_tot, std::plus<double>());
        return s_tot;
    }

    template <class T>
    void parallel_buff_3D<T>::FFT3D(const std::array<T, 3> e, const T _1)
    {
        // FFT on z
        for (size_t i = 0; i < N_loc[0]; ++i)
            for (size_t j = 0; j < N_loc[1]; ++j)
                FFT(&(*this)(i, j, 0), &(*this)(i, j + 1, 0), e[2], _1);

        // FFT on y
        transpose_yz();
        for (size_t i = 0; i < N_loc[0]; ++i)
            for (size_t j = 0; j < N_loc[1]; ++j)
                FFT(&(*this)(i, j, 0), &(*this)(i, j + 1, 0), e[1], _1);
        transpose_yz();

        // FFT on x
        transpose_xz();
        for (size_t i = 0; i < N_loc[0]; ++i)
            for (size_t j = 0; j < N_loc[1]; ++j)
                FFT(&(*this)(i, j, 0), &(*this)(i, j + 1, 0), e[0], _1);
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
