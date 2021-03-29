for directory in C1 Cs; do
    echo
    echo "Entre abinitio/test/"$directory
    cd abinitio/test/$directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../../../..
done

for directory in obnet Hderiva; do
    echo
    echo "Entre "$directory"/test"
    cd $directory/test/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../../..
done
