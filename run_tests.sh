#!/bin/bash

PROJECT_ROOT=$(pwd)
BUILD_DIR=$PROJECT_ROOT/build
OUTPUT_FILE_NATIVE=$BUILD_DIR/native.csv
OUTPUT_FILE_VERLET=$BUILD_DIR/verlet.csv

cd $BUILD_DIR

# Tests are carried out by varying the number of processors, and the level of mesh refinement
PROCESSORS=(1)
MESH_REFINEMENT=(3 4 5)

touch $OUTPUT_FILE_NATIVE
echo "processors,mesh_refinement,dofs,time" > $OUTPUT_FILE_NATIVE

# Run the tests
for p in ${PROCESSORS[@]}; do
    for m in ${MESH_REFINEMENT[@]}; do
        echo "Running test with $p processors and $m mesh refinements"
        mpirun -np $p ./VerletParallel $m > temp
        

        # Extract the number of degrees of freedom and the time taken
        dofs=$(grep "Number of degrees of freedom" temp | awk '{print $NF}')
        echo "dofs: $dofs"
        
        time=$(grep "Total wallclock time elapsed since start" temp | awk -F'|' '{print $3}' | xargs)
        echo "time: $time"
        # Cancel all outputs
        rm *tu
    done
done