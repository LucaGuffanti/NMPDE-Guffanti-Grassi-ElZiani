### Compiling and Executing the Code
> [!CAUTION]
> The project is implemented with deal.II 9.3.1, make sure to have the 2022 mk modules installed.

To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
You can choose to include some tests in the compilation by modifying the `CMakeLists.txt` file that's present in the root of the repository.
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
All the executables will be created in the `build` directory. By default, executables are configured to access the first command line parameter which should
contain the number of mesh refinement cycles to perform on the standard mesh we use (1x1 square with center at 0.5, 0.5). If no additional input is provided (`argc<2`), then the program will halt and print an error message. To run the code use the following command:
```bash
$ ./executable <number_of_mesh_refinements>
```
The executables are the following
- `WaveParallel` Solves the problem using the $\theta$-method (defaults to Crank-Nicolson). Parallelized.
- `WaveSerial` Solves the problem using the $\theta$-method (defaults to Crank-Nicolson). Serial.
- `VerletParallel` Solves the problem using Verlet integration. Parallelized.
- `VerletSerial` Solves the problem using Verlet integration. Serial.

Please note that the code supports the use of a mesh from an external input. If necessary, construct the solver objects by passing the path to the mesh file as a parameter, and rebuild the binaries.

### Visualizing the Output
All outputs will be written in the directory from which the binary is called (generally `build/`). For visualization purposes import the files in ParaView and proceed from there.
