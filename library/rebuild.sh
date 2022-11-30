for directory in abinitio SASDIC DimRed obnet Hderiva; do
    echo
    echo "Entre "$directory
    cd $directory/build
    cmake --build .
    cd ../..
done
