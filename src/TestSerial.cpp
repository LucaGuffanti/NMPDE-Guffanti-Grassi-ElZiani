#include <iostream>
#include "WaveSerial.hpp"

int main()
{
    const unsigned int degree = 1;
    const double interval = 5.0;
    const double time_step = 1./64;
    const double theta = 0.5;

    WaveEquationSerial<2> wave_eq(
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