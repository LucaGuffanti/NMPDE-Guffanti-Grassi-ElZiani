#include <iostream>
#include "WaveSerial.hpp"

int main()
{
    const unsigned int degree = 2;
    const double interval = 10.0;
    const double time_step = 1./128;
    const double theta = 0.5;

    WaveEquationSerial<2> wave_eq(
        degree,
        interval,
        time_step,
        theta
    );

    wave_eq.setup(7);
    wave_eq.assemble_matrices();
    wave_eq.run();

    return 0;
}