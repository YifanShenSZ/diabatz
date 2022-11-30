echo "Entre libHd"
cd libHd/build
cmake --build .
cd ../..

for directory in bin2txt txt2bin rand-txt data-stats vibronics; do
    echo
    echo "Entre "$directory
    cd $directory/build
    cmake --build .
    cd ../..
done
