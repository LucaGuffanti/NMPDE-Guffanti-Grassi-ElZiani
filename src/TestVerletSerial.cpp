#include <iostream>
#include "VerletSerial.hpp"

int main()
{
    const unsigned int degree = 1;
    const double interval = 3.0;
    const double time_step = 1./128;

    VerletSerial<2> wave_eq(
        degree,
        interval,
        time_step
    );

    wave_eq.setup();
    wave_eq.assemble_matrices(false);
    wave_eq.run();


    return 0;
}