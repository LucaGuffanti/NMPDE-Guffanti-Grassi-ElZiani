#include <iostream>
#include "WaveParallel.hpp"

int main(int argc, char *argv[])
{

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);    

    const unsigned int degree = 1;
    const double interval = 15.0;
    const double time_step = 1./16;
    const double theta = 0.5;

    WaveEquationParallel<2> wave_eq(
        degree,
        interval,
        time_step,
        theta
    );

    wave_eq.setup();
    wave_eq.assemble_matrices(false);
    wave_eq.run();


    return 0;
}