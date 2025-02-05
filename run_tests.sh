#!/bin/bash

PROJECT_ROOT=$(pwd)
BUILD_DIR=$PROJECT_ROOT/build
OUTPUT_FILE_NATIVE=$BUILD_DIR/native.csv
OUTPUT_FILE_VERLET=$BUILD_DIR/verlet.csv

cd $BUILD_DIR

# Tests are carried out by varying the number of processors, and the level of mesh refinement
PROCESSORS=(1 2 4 8 16 32 64 128)
MESH_REFINEMENT=(2 3 4 5 6 7 8)

touch $OUTPUT_FILE_VERLET
echo "processors,mesh_refinement,dofs,time" > $OUTPUT_FILE_VERLET

# Run the tests
for p in ${PROCESSORS[@]}; do
    for m in ${MESH_REFINEMENT[@]}; do
        echo "Running test with $p processors and $m mesh refinements"
        mpirun -np $p ./VerletParallel $m > temp
        cat temp > "verlet_p${p}_m${m}.txt"

        # Extract the number of degrees of freedom and the time taken
        dofs=$(grep "Number of degrees of freedom" temp | awk '{print $NF}')
        echo "dofs: $dofs"
        
        time=$(grep "Total wallclock time elapsed since start" temp | awk -F'|' '{print $3}' | xargs)
        echo "time: $time"
        # Cancel all outputs

        echo "$p,$m,$dofs,$time" >> $OUTPUT_FILE_VERLET

        tail -n 20 temp
        
        rm *tu
    done
done