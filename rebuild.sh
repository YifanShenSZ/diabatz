for directory in v0.0.0 v0.0.1 v0.0.2; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm *.exe
    cmake --build .
    cd ../..
done
