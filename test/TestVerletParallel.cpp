#include <iostream>
#include "VerletParallel.hpp"

int main(int argc, char **argv)
{

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);    

    const unsigned int degree = 2;
    const double interval = 1.0;
    const double time_step = 1./2048;

    VerletParallel<2> wave_eq(
        degree,
        interval,
        time_step
    );

    const unsigned int times = atoi(argv[1]);

    wave_eq.setup(times);
    wave_eq.assemble_matrices();
    wave_eq.run();


    return 0;
}