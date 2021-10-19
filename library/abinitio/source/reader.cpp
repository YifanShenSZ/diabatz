#include <CppLibrary/utility.hpp>

#include <tchem/chemistry.hpp>

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
    if (user_list.empty()) throw std::invalid_argument(
    "abinitio::Reader::Reader: User should specify files or directories");
    for (const std::string & item : user_list) {
        if (item.back() == '/') data_directories_.push_back(item);
        else {
            std::string prefix = CL::utility::GetPrefix(item);
            std::ifstream ifs; ifs.open(item);
            if (! ifs.good()) throw CL::utility::file_error(item);
            while (true) {
                std::string directory;
                ifs >> directory;
                if (! ifs.good()) break;
                if (directory.back() != '/') directory += "/";
                directory = prefix + directory;
                data_directories_.push_back(directory);
            }
            ifs.close();
        }
    }
}
Reader::~Reader() {}

const std::vector<std::string> & Reader::data_directories() const {return data_directories_;}

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
// Number of data points in this directory
size_t Reader::NData(const std::string & data_directory) const {
    return CL::utility::NLines(data_directory + "energy.data");
}
// Number of atoms constituting the molecule
size_t Reader::NAtoms() const {
    size_t NAtoms_ = CL::utility::NLines(data_directories_[0] + "geom.data")
                   / CL::utility::NLines(data_directories_[0] + "energy.data");
    return NAtoms_;
}
// Number of electronic states in this directory
size_t Reader::NStates(const std::string & data_directory) const {
    std::ifstream ifs; ifs.open(data_directory + "energy.data");
        std::string line;
        std::getline(ifs, line);
        std::vector<std::string> strs = CL::utility::split(line);
    ifs.close();
    return strs.size();
}

// Read geometries
std::shared_ptr<DataSet<Geometry>> Reader::read_GeomSet() const {
    std::vector<std::shared_ptr<Geometry>> pgeoms;
    for (const std::string & data_directory : data_directories_) {
        std::vector<GeomLoader> loaders(NData(data_directory));
        for (auto & loader : loaders) loader.reset(3 * NAtoms());
        load_weight(loaders, data_directory);
        load_geom  (loaders, data_directory);
        for (auto & loader : loaders) pgeoms.push_back(std::make_shared<Geometry>(loader));
    }
    std::shared_ptr<DataSet<Geometry>> GeomSet = std::make_shared<DataSet<Geometry>>(pgeoms);
    return GeomSet;
}

// Read Hamiltonians
std::tuple<std::shared_ptr<DataSet<RegHam>>, std::shared_ptr<DataSet<DegHam>>>
Reader::read_HamSet() const {
    std::vector<std::shared_ptr<RegHam>> pregs;
    std::vector<std::shared_ptr<DegHam>> pdegs;
    for (const std::string & data_directory : data_directories_) {
        std::vector<HamLoader> loaders(NData(data_directory));
        for (auto & loader : loaders) loader.reset(3 * NAtoms(), NStates(data_directory));
        load_weight(loaders, data_directory);
        load_geom(loaders, data_directory);
        load_energy(loaders, data_directory);
        load_dH(loaders, data_directory);
        for (auto & loader : loaders) {
            if (tchem::chem::check_degeneracy(loader.energy, deg_thresh_)) {
                pdegs.push_back(std::make_shared<DegHam>(loader));
            } else {
                pregs.push_back(std::make_shared<RegHam>(loader));
            }
        }
    }
    std::shared_ptr<DataSet<RegHam>> RegSet = std::make_shared<DataSet<RegHam>>(pregs);
    std::shared_ptr<DataSet<DegHam>> DegSet = std::make_shared<DataSet<DegHam>>(pdegs);
    return std::make_tuple(RegSet, DegSet);
}

} // namespace abinitio