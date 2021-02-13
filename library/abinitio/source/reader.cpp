#include <CppLibrary/utility.hpp>

#include <tchem/chemistry.hpp>

#include <abinitio/DataSet.hpp>
#include <abinitio/geometry.hpp>
#include <abinitio/Hamiltonian.hpp>

#include <abinitio/reader.hpp>

namespace abinitio {

Reader::Reader() {}
// User specifies a list of files or directories for data sets
// This constructor assumes:
//     A file is intented to contain a (long) list of directories
//     A directory is who truely holds a data set
// This constructor verifies if user inputs are directories (end with /),
// otherwise files then read them for directories
Reader::Reader(const std::vector<std::string> & user_list) {
    for (std::string item : user_list) {
        if (item.back() == '/') data_directories_.push_back(item);
        else {
            std::string prefix = CL::utility::GetPrefix(item);
            std::ifstream ifs; ifs.open(item);
                while (ifs.good()) {
                    std::string directory;
                    ifs >> directory;
                    if (directory.back() != '/') directory += "/";
                    directory = prefix + directory;
                    data_directories_.push_back(directory);
                }
            ifs.close();
        }
    }
}
Reader::~Reader() {}

std::vector<std::string> Reader::data_directories() const {return data_directories_;}

void Reader::pretty_print(std::ostream & stream) const {
    stream << "The data set will be read from: \n    ";
    size_t line_length = 4;
    for (std::string directory : data_directories_) {
        line_length += directory.size() + 2;
        if (line_length > 75) {
            stream << '\n' << "    ";
            line_length = 4;
        }
        stream << directory << ", ";
    }
    stream << '\n';
}
// Number of data points per directory
std::vector<size_t> Reader::NData() const {
    std::vector<size_t> NData_(data_directories_.size());
    for (size_t i = 0; i < data_directories_.size(); i++)
    NData_[i] = CL::utility::NLines(data_directories_[i] + "energy.data");
    return NData_;
}
// Number of atoms constituting the molecule
size_t Reader::NAtoms() const {
    size_t NAtoms_ = CL::utility::NLines(data_directories_[0] + "geom.data")
                   / CL::utility::NLines(data_directories_[0] + "energy.data");
    return NAtoms_;
}

// Read geometries
std::shared_ptr<DataSet<Geometry>> Reader::read_GeomSet() const {
    // Prepare
    std::vector<size_t> NData_PerDir = NData();
    size_t NData_total = std::accumulate(NData_PerDir.begin(), NData_PerDir.end(), 0);
    size_t NAtoms_ = NAtoms();
    // Read data files
    std::vector<std::shared_ptr<Geometry>> pgeoms(NData_total);
    size_t count = 0;
    for (size_t id = 0; id < data_directories_.size(); id++) {
        // Read geometries in to loaders
        std::vector<GeomLoader> loaders(NData_PerDir[id]);
        for (auto & loader : loaders) loader.reset(3 * NAtoms_);
        std::ifstream ifs; ifs.open(data_directories_[id] + "geom.data");
            for (auto & loader : loaders)
            for (size_t i = 0; i < NAtoms_; i++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; loader.geom[3 * i    ] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * i + 1] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * i + 2] = dbletemp;
            }
        ifs.close();
        // Insert to data set
        for (auto & loader : loaders) {
            pgeoms[count] = std::make_shared<Geometry>(loader);
            count++;
        }
    }
    // Create DataSet with data set loader
    std::shared_ptr<DataSet<Geometry>> GeomSet = std::make_shared<DataSet<Geometry>>(pgeoms);
    return GeomSet;
}

// Read Hamiltonians
std::tuple<std::shared_ptr<DataSet<RegHam>>, std::shared_ptr<DataSet<DegHam>>>
Reader::read_HamSet() const {
    // Prepare
    std::vector<size_t> NData_PerDir = NData();
    size_t NData_total = std::accumulate(NData_PerDir.begin(), NData_PerDir.end(), 0);
    size_t NAtoms_ = NAtoms();
    // Read data files
    std::vector<std::shared_ptr<RegHam>> pregs(NData_total);
    std::vector<std::shared_ptr<DegHam>> pdegs(NData_total);
    size_t NRegData = 0;
    size_t NDegData = 0;
    for (size_t id = 0; id < data_directories_.size(); id++) {
        // for file input
        std::ifstream ifs;
        std::string line;
        std::vector<std::string> strs;
        // Infer number of electronic states
        ifs.open(data_directories_[id] + "energy.data");
        std::getline(ifs, line);
        ifs.close();
        CL::utility::split(line, strs);
        int64_t NStates = strs.size();
        // Read data in to loaders
        std::vector<HamLoader> loaders(NData_PerDir[id]);
        for (auto & loader : loaders) loader.reset(3 * NAtoms_, NStates);
        // geometry
        ifs.open(data_directories_[id] + "geom.data");
            for (auto & loader : loaders)
            for (size_t j = 0; j < NAtoms_; j++) {
                std::string line; ifs >> line;
                double dbletemp;
                ifs >> dbletemp; loader.geom[3 * j    ] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * j + 1] = dbletemp;
                ifs >> dbletemp; loader.geom[3 * j + 2] = dbletemp;
            }
        ifs.close();
        // energy
        ifs.open(data_directories_[id] + "energy.data");
            for (auto & loader : loaders) {
                std::getline(ifs, line);
                CL::utility::split(line, strs);
                for (size_t j = 0; j < NStates; j++)
                loader.energy[j] = std::stod(strs[j]);
            }
        ifs.close();
        // gradient
        for (size_t istate = 0; istate < NStates; istate++) {
            ifs.open(data_directories_[id] + "cartgrad-" + std::to_string(istate+1) + ".data");
                for (auto & loader : loaders)
                for (size_t j = 0; j < 3 * NAtoms_; j++) {
                    double dbletemp; ifs >> dbletemp;
                    loader.dH[istate][istate][j] = dbletemp;
                }
            ifs.close();
        for (size_t jstate = istate + 1; jstate < NStates; jstate++) {
            ifs.open(data_directories_[id] + "cartgrad-" + std::to_string(istate+1) + "-" + std::to_string(jstate+1) + ".data");
                for (auto & loader : loaders)
                for (size_t j = 0; j < 3 * NAtoms_; j++) {
                    double dbletemp; ifs >> dbletemp;
                    loader.dH[istate][jstate][j] = dbletemp;
                }
            ifs.close();
        } }
        // Insert to data set loader
        for (auto & loader : loaders) {
            if (tchem::chem::check_degeneracy(loader.energy, deg_thresh_)) {
                pdegs[NDegData] = std::make_shared<DegHam>(loader);
                NDegData++;
            } else {
                pregs[NRegData] = std::make_shared<RegHam>(loader);
                NRegData++;
            }
        }
    }
    // Create DataSet with data set loader
    pregs.resize(NRegData);
    pdegs.resize(NDegData);
    std::shared_ptr<DataSet<RegHam>> RegSet = std::make_shared<DataSet<RegHam>>(pregs);
    std::shared_ptr<DataSet<DegHam>> DegSet = std::make_shared<DataSet<DegHam>>(pdegs);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace abinitio