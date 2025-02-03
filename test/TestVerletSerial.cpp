#include <iostream>
#include "VerletSerial.hpp"

int main()
{
    const unsigned int degree = 2;
    const double interval = 10.0;
    const double time_step = 1./256;

    VerletSerial<2> wave_eq(
        degree,
        interval,
        time_step
    );

    wave_eq.setup(5);
    wave_eq.assemble_matrices();
    wave_eq.run();

    return 0;
}