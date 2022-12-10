#ifndef abinitio_reader_hpp
#define abinitio_reader_hpp

#include <abinitio/geometry.hpp>
#include <abinitio/energy.hpp>
#include <abinitio/Hamiltonian.hpp>
#include <abinitio/DataSet.hpp>

namespace abinitio {

class Reader {
    protected:
        double deg_thresh_;
        std::vector<std::string> data_directories_;
    public:
        Reader();
        // User specifies a list of files or directories for data sets
        // This constructor assumes:
        //     A file is intented to contain a (long) list of directories
        //     A directory is who truely holds a data set
        // This constructor verifies if user inputs are directories (end with /),
        // otherwise files then read them for directories
        Reader(const std::vector<std::string> & user_list, const double& _deg_thresh=0.0001);
        ~Reader();

        const std::vector<std::string> & data_directories() const;

        void pretty_print(std::ostream & stream) const;
        // number of data points per directory
        std::vector<size_t> NData() const;
        // number of data points in this directory
        size_t NData(const std::string & data_directory) const;
        // number of atoms constituting the molecule
        size_t NAtoms() const;
        // number of electronic states in this directory
        size_t NStates(const std::string & data_directory) const;

        template <typename T> void load_weight(std::vector<T> & loaders, const std::string & data_directory) const {
            std::string file = data_directory + "weight.txt";
            std::ifstream ifs; ifs.open(file);
            if (! ifs.good()) throw CL::utility::file_error(file);
            for (T & loader : loaders) ifs >> loader.weight;
            ifs.close();
        }
        template <typename T> void load_geom(std::vector<T> & loaders, const std::string & data_directory) const {
            std::string file = data_directory + "geom.data";
            std::ifstream ifs; ifs.open(file);
            if (! ifs.good()) throw CL::utility::file_error(file);
            for (T & loader : loaders)
            for (size_t i = 0; i < loader.geom.numel() / 3; i++) {
                std::string symbol; ifs >> symbol;
                double dbletemp;
                ifs >> dbletemp; loader.geom[3 * i    ].fill_(dbletemp);
                ifs >> dbletemp; loader.geom[3 * i + 1].fill_(dbletemp);
                ifs >> dbletemp; loader.geom[3 * i + 2].fill_(dbletemp);
            }
            ifs.close();
        }
        template <typename T> void load_energy(std::vector<T> & loaders, const std::string & data_directory) const {
            std::string file = data_directory + "energy.data";
            std::ifstream ifs; ifs.open(file);
            if (! ifs.good()) throw CL::utility::file_error(file);
            for (T & loader : loaders)
            for (size_t j = 0; j < loader.energy.size(0); j++) {
                double dbletemp; ifs >> dbletemp;
                loader.energy[j].fill_(dbletemp);
            }
            ifs.close();
        }
        template <typename T> void load_dH(std::vector<T> & loaders, const std::string & data_directory) const {
            for (size_t istate = 0; istate < loaders[0].dH.size(0); istate++) {
                std::string file = data_directory + "cartgrad-" + std::to_string(istate + 1) + ".data";
                std::ifstream ifs; ifs.open(file);
                if (! ifs.good()) throw CL::utility::file_error(file);
                for (T & loader : loaders)
                for (size_t j = 0; j < loader.geom.numel(); j++) {
                    double dbletemp; ifs >> dbletemp;
                    loader.dH[istate][istate][j].fill_(dbletemp);
                }
                ifs.close();
                for (size_t jstate = istate + 1; jstate < loaders[0].dH.size(1); jstate++) {
                    std::string file = data_directory + "cartgrad-" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + ".data";
                    std::ifstream ifs; ifs.open(file);
                    if (! ifs.good()) throw CL::utility::file_error(file);
                    for (T & loader : loaders)
                    for (size_t j = 0; j < loader.geom.numel(); j++) {
                        double dbletemp; ifs >> dbletemp;
                        loader.dH[istate][jstate][j].fill_(dbletemp);
                    }
                    ifs.close();
                }
            }
        }

        // read geometries
        std::shared_ptr<DataSet<Geometry>> read_GeomSet() const;
        // read energies
        std::shared_ptr<DataSet<Energy>> read_EnergySet() const;
        // read Hamiltonians
        std::tuple<std::shared_ptr<DataSet<RegHam>>, std::shared_ptr<DataSet<DegHam>>> read_HamSet() const;
};

} // namespace abinitio

#endif