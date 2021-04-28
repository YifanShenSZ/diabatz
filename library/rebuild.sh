for directory in abinitio DimRed obnet Hderiva; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f lib*
    cmake --build .
    cd ../..
done