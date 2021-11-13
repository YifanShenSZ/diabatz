for directory in abinitio SASIC DimRed obnet Hderiva; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm lib*
    cmake --build .
    cd ../..
done
