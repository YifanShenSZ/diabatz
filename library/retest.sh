echo "Entre abinitio/test/"
cd abinitio/test/
bash retest.sh
cd ../..

for directory in SASDIC DimRed obnet Hderiva; do
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
