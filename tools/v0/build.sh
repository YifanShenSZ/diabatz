echo "Entre libHd"
cd libHd
# build
if [ -d build ]; then rm -r build; fi
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort ..
cmake --build .
cd ..
# create lib/
if [ -d lib ]; then rm -r lib; fi
mkdir lib
cd lib
ln -s ../build/libHd.a
# finish
cd ../..

for directory in bin2txt txt2bin rand-txt data-stats vibronics; do
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
    # finish
    cd ..
done
