#include <iostream>
#include "VerletSerial.hpp"

int main(int argc, char** argv)
{
    const unsigned int degree = 2;
    const double interval = 10.0;
    const double time_step = 1./256;

    VerletSerial<2> wave_eq(
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