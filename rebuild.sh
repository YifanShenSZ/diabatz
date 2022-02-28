for directory in \
v0.0.0 v0.0.1 v0.0.2 v0.1.0 v0.2.0 v0.3.0 v0.3.2 \
v1.0.0 v1.1.0 v1.3.0 v1.3.2; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm *.exe
    cmake --build .
    cd ../..
done
