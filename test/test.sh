for directory in abinitio SAabinitio obnet Hderiva; do
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done