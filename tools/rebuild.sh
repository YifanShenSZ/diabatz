cd v0
bash rebuild.sh
cd ..

cd v1
bash rebuild.sh
cd ..

for directory in CNPI2point cart2SASDIC; do
    echo
    echo "Entre "$directory
    cd $directory/build
    cmake --build .
    cd ../..
done

for directory in eval Hessian RMSD critics vibration; do
    echo
    echo "Entre "$directory
    cd $directory
    # v0
    cd v0
    cmake --build .
    cd ..
    # v1
    cd v1
    cmake --build .
    cd ..
    # leave directory
    cd ..
done
