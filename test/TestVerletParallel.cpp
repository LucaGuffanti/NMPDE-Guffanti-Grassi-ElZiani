#include <iostream>
#include "VerletParallel.hpp"

int main(int argc, char *argv[])
{

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);    

    const unsigned int degree = 2;
    const double interval = 10.0;
    const double time_step = 1./256;

    VerletParallel<2> wave_eq(
        degree,
        interval,
        time_step
    );

    wave_eq.setup(5);
    wave_eq.assemble_matrices();
    wave_eq.run();


    return 0;
}