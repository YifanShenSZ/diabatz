for directory in abinitio obnet Hderiva; do
    cd $directory/build
    rm -f lib*
    cmake --build .
    cd ../..
done