cd v0
bash build.sh
cd ..

cd v1
bash build.sh
cd ..

for directory in CNPI2point cart2SASDIC; do
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
    if [ -f $directory.exe ]; then rm $directory.exe; fi
    ln -s build/$directory.exe
    cd ..
done

for directory in eval Hessian RMSD critics vibration; do
    echo
    echo "Entre "$directory
    cd $directory
    # v0
    if [ -d v0 ]; then rm -r v0; fi
    mkdir v0
    cd v0
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DHd_DIR=~/Software/Mine/diabatz/tools/v0/libHd/share/cmake/Hd ..
    cmake --build .
    cd ..
    # v1
    if [ -d v1 ]; then rm -r v1; fi
    mkdir v1
    cd v1
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort -DHd_DIR=~/Software/Mine/diabatz/tools/v1/libHd/share/cmake/Hd ..
    cmake --build .
    cd ..
    # leave directory
    cd ..
done
