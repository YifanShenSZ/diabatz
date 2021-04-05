echo "Entre libHd"
cd libHd/build
rm -f lib*
cmake --build .
cd ../..

for directory in bin2txt txt2bin vibronics; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f $directory.exe
    cmake --build .
    cd ../..
done