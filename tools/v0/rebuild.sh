cd libHd/build
rm -f lib*
cmake --build .
cd ../..

for directory in eval RMSD bin2txt; do
    cd $directory/build
    rm -f $directory.exe
    cmake --build .
    cd ../..
done