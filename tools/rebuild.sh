cd v0
bash rebuild.sh
cd ..

cd v1
bash rebuild.sh
cd ..

for directory in CNPI2point autoencoder cart2SASDIC; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm *.exe
    cmake --build .
    cd ../..
done

for directory in eval Hessian RMSD critics vibration; do
    echo
    echo "Entre "$directory
    cd $directory
    # v0
    cd v0
    rm *.exe
    cmake --build .
    cd ..
    # v1
    cd v1
    rm *.exe
    cmake --build .
    cd ..
    # leave directory
    cd ..
done
