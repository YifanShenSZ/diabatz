for directory in abinitio obnet Hderiva; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f lib*
    cmake --build .
    cd ../..
done