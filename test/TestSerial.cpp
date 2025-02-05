#include <iostream>
#include "WaveSerial.hpp"

int main(int argc, char** argv)
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

    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <times>" << std::endl;
        return 1;
    }
    const unsigned int times = atoi(argv[1]);

    wave_eq.setup(times);
    wave_eq.assemble_matrices();
    wave_eq.run();

    return 0;
}