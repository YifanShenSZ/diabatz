echo "Entre libHd"
cd libHd/build
rm lib*
cmake --build .
cd ../..

for directory in bin2txt txt2bin vibronics; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm *.exe
    cmake --build .
    cd ../..
done
