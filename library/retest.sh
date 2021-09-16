for directory in C1 Cs; do
    echo
    echo "Entre abinitio/test/"$directory
    cd abinitio/test/$directory/build
    rm test.exe
    cmake --build .
    cd ../input
    ../build/test.exe
    cd ../../../..
done

for directory in DimRed obnet Hderiva; do
    echo
    echo "Entre "$directory"/test"
    cd $directory/test/build
    rm test.exe
    cmake --build .
    cd ..
    if [ -d input ]; then
        cd input
        ../build/test.exe
        cd ../../..
    else
       ./build/test.exe
       cd ../..
    fi
done
