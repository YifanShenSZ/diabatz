#ifndef abinitio_reader_hpp
#define abinitio_reader_hpp

#include <abinitio/geometry.hpp>
#include <abinitio/Hamiltonian.hpp>
#include <abinitio/DataSet.hpp>

namespace abinitio {

class Reader {
    protected:
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
        // Number of data points in this directory
        size_t NData(const std::string & data_directory) const;
        // Number of atoms constituting the molecule
        size_t NAtoms() const;
        // Number of electronic states in this directory
        size_t NStates(const std::string & data_directory) const;

        template <typename T> void load_geom(std::vector<T> & loaders, const std::string & data_directory) const {
            std::ifstream ifs; ifs.open(data_directory + "geom.data");
            assert((data_directory + "geom.data" + " must be good", ifs));
            for (T & loader : loaders)
            for (size_t i = 0; i < loader.geom.numel() / 3; i++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; loader.geom[3 * i    ] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * i + 1] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * i + 2] = dbletemp;
            }
            ifs.close();
        }
        template <typename T> void load_energy(std::vector<T> & loaders, const std::string & data_directory) const {
            std::ifstream ifs; ifs.open(data_directory + "energy.data");
            assert((data_directory + "energy.data" + " must be good", ifs));
            for (T & loader : loaders) {
                std::string line;
                std::getline(ifs, line);
                std::vector<std::string> strs = CL::utility::split(line);
                for (size_t j = 0; j < loader.energy.size(0); j++)
                loader.energy[j] = std::stod(strs[j]);
            }
            ifs.close();
        }
        template <typename T> void load_dH(std::vector<T> & loaders, const std::string & data_directory) const {
            for (size_t istate = 0; istate < loaders[0].dH.size(0); istate++) {
                std::ifstream ifs; ifs.open(data_directory + "cartgrad-" + std::to_string(istate+1) + ".data");
                assert((data_directory + "cartgrad-" + std::to_string(istate+1) + ".data" + " must be good", ifs));
                for (T & loader : loaders)
                for (size_t j = 0; j < loader.geom.numel(); j++) {
                    double dbletemp; ifs >> dbletemp;
                    loader.dH[istate][istate][j] = dbletemp;
                }
                ifs.close();
            for (size_t jstate = istate + 1; jstate < loaders[0].dH.size(1); jstate++) {
                std::ifstream ifs; ifs.open(data_directory + "cartgrad-" + std::to_string(istate+1) + "-" + std::to_string(jstate+1) + ".data");
                assert((data_directory + "cartgrad-" + std::to_string(istate+1) + "-" + std::to_string(jstate+1) + ".data" + " must be good", ifs));
                for (T & loader : loaders)
                for (size_t j = 0; j < loader.geom.numel(); j++) {
                    double dbletemp; ifs >> dbletemp;
                    loader.dH[istate][jstate][j] = dbletemp;
                }
                ifs.close();
            } }
        }

        // Read geometries
        std::shared_ptr<DataSet<Geometry>> read_GeomSet() const;
        // Read Hamiltonians
        std::tuple<std::shared_ptr<DataSet<RegHam>>, std::shared_ptr<DataSet<DegHam>>> read_HamSet() const;
};

} // namespace abinitio

#endif