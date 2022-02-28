for directory in \
v0.0.0 v0.0.1 v0.0.2 v0.1.0 v0.2.0 v0.3.0 v0.3.2 \
v1.0.0 v1.1.0 v1.3.0 v1.3.2; do
    echo
    echo "Entre "$directory
    cd $directory
    # build
    if [ -d build ]; then rm -r build; fi
    mkdir build
    cd build
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort ..
    cmake --build .
    cd ..
    # link exe
    if [ -f diabatz.exe ]; then rm diabatz.exe; fi
    ln -s build/diabatz.exe
    # finish
    cd ..
done
