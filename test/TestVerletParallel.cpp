#include <iostream>
#include "VerletParallel.hpp"

int main(int argc, char *argv[])
{

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);    

    const unsigned int degree = 1;
    const double interval = 5.0;
    const double time_step = 1./128;

    VerletParallel<2> wave_eq(
        degree,
        interval,
        time_step
    );

    // wave_eq.setup("/home/lucaguf/Documenti/dev/NMPDE-Guffanti-Grassi-ElZiani/mesh/mesh-square-h0.100000.msh");
    wave_eq.setup("/home/lucaguf/Documenti/dev/NMPDE-Guffanti-Grassi-ElZiani/mesh/mesh-square-h0.050000.msh");
    wave_eq.assemble_matrices(false);
    wave_eq.run();


    return 0;
}