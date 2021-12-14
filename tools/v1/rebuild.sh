echo "Entre libHd"
cd libHd/build
rm lib*
cmake --build .
cd ../..

for directory in BLHessian; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm *.exe
    cmake --build .
    cd ../..
done
