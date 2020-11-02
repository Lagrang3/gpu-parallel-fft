#define BOOST_TEST_MODULE parallel_sum
#include <boost/mpi.hpp>
#include <boost/test/unit_test.hpp>
#include <gpfft/parallel_buffer.hpp>

namespace ut = boost::unit_test;
namespace mpi = boost::mpi;

struct fixture
{
    fixture() {}

    ~fixture() {}

    static mpi::environment env;
    static mpi::communicator world;
};

mpi::environment fixture::env;
mpi::communicator fixture::world;

BOOST_TEST_GLOBAL_FIXTURE(fixture);

BOOST_AUTO_TEST_CASE(tanspose_xz)
{
    BOOST_REQUIRE(fixture::world.size() == 2);
    const int nc = 4;
    gpfft::parallel_buff_3D<int> A(fixture::world, {nc, nc, nc});

    std::valarray<int> R0{0x000, 0x001, 0x002, 0x003, 0x010, 0x011, 0x012,
                          0x013, 0x020, 0x021, 0x022, 0x023, 0x030, 0x031,
                          0x032, 0x033, 0x100, 0x101, 0x102, 0x103, 0x110,
                          0x111, 0x112, 0x113, 0x120, 0x121, 0x122, 0x123,
                          0x130, 0x131, 0x132, 0x133},
        R1{0x200, 0x201, 0x202, 0x203, 0x210, 0x211, 0x212, 0x213,
           0x220, 0x221, 0x222, 0x223, 0x230, 0x231, 0x232, 0x233,
           0x300, 0x301, 0x302, 0x303, 0x310, 0x311, 0x312, 0x313,
           0x320, 0x321, 0x322, 0x323, 0x330, 0x331, 0x332, 0x333};

    std::valarray<int> tmp = fixture::world.rank() == 0 ? R0 : R1;

    BOOST_REQUIRE(tmp.size() == A.size());

    for (size_t i = 0; i < tmp.size(); ++i)
        A[i] = tmp[i];

    const int parallel_sum = A.sum();
    const int serial_sum = R0.sum() + R1.sum();
    BOOST_CHECK(parallel_sum == serial_sum);
}
