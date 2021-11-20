cd v0
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

for directory in eval RMSD critics vibration; do
    echo
    echo "Entre "$directory
    cd $directory/v0
    rm *.exe
    cmake --build .
    cd ../..
done
