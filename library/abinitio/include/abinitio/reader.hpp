#ifndef abinitio_reader_hpp
#define abinitio_reader_hpp

#include <abinitio/geometry.hpp>
#include <abinitio/Hamiltonian.hpp>
#include <abinitio/DataSet.hpp>

namespace abinitio {

class Reader {
    private:
        double deg_thresh_ = 0.0001;
        std::vector<std::string> data_directories_;
    public:
        Reader();
        // User specifies a list of files or directories for data sets
        // This constructor assumes:
        //     A file is intented to contain a (long) list of directories
        //     A directory is who truely holds a data set
        // This constructor verifies if user inputs are directories (end with /),
        // otherwise files then read them for directories
        Reader(const std::vector<std::string> & user_list);
        ~Reader();

        std::vector<std::string> data_directories() const;

        void pretty_print(std::ostream & stream) const;
        // Number of data points per directory
        std::vector<size_t> NData() const;
        // Number of atoms constituting the molecule
        size_t NAtoms() const;

        // Read geometries
        std::shared_ptr<DataSet<Geometry>> read_GeomSet() const;
        // Read Hamiltonians
        std::tuple<std::shared_ptr<DataSet<RegHam>>, std::shared_ptr<DataSet<DegHam>>> read_HamSet() const;
};

} // namespace abinitio

#endif