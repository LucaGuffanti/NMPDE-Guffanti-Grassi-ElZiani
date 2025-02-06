#include <iostream>
#include "WaveParallel.hpp"

int main(int argc, char **argv)
{

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);    

    const unsigned int degree = 2;
    const double interval = 10.0;
    const double time_step = 1./64;
    const double theta = 0.5;

    WaveEquationParallel<2> wave_eq(
        degree,
        interval,
        time_step,
        theta
    );
    
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <times>" << std::endl;
        return 1;
    }
    const unsigned int times = atoi(argv[1]);

    wave_eq.setup(times);
    wave_eq.assemble_matrices();
    wave_eq.run();

    wave_eq.print_timer_data();
    return 0;
}