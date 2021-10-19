for directory in C1 Cs C2v; do
    echo
    echo "Entre abinitio/test/"$directory
    cd abinitio/test/$directory
    # build
    if [ -d build ]; then rm -r build; fi
    mkdir build
    cd build
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort ..
    cmake --build .
    cd ..
    # run
    if [ -d input ]; then
        cd input
        ../build/test.exe
        cd ../../../..
    else
       ./build/test.exe
       cd ../../..
    fi
done

for directory in DimRed obnet Hderiva; do
    echo
    echo "Entre "$directory"/test"
    cd $directory/test
    # build
    if [ -d build ]; then rm -r build; fi
    mkdir build
    cd build
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort ..
    cmake --build .
    cd ..
    # run
    if [ -d input ]; then
        cd input
        ../build/test.exe
        cd ../../..
    else
       ./build/test.exe
       cd ../..
    fi
done
