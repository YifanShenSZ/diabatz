#ifndef abinitio_SAreader_hpp
#define abinitio_SAreader_hpp

#include <glob.h>

#include <abinitio/reader.hpp>
#include <abinitio/SAgeometry.hpp>
#include <abinitio/SAHamiltonian.hpp>
#include <abinitio/DataSet.hpp>

namespace abinitio {

class SAReader : public Reader {
    private:
        std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*cart2CNPI_)(const at::Tensor &);
    public:
        SAReader();
        // See the base class constructor for details of `user_list`
        // `cart2CNPI` takes in Cartesian coordinate r, returns CNPI group symmetry adapted internal coordinates and their Jacobians over r
        SAReader(const std::vector<std::string> & user_list,
                 std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>> (*_cart2CNPI)(const at::Tensor &));
        ~SAReader();

        template <typename T> void load_CNPI2point(std::vector<T> & loaders, const std::string & data_directory) const {
            std::string file = data_directory + "CNPI2point.txt";
            std::ifstream ifs; ifs.open(file);
            if (! ifs.good()) throw CL::utility::file_error(file);
            for (auto & loader : loaders) {
                std::string line;
                std::getline(ifs, line);
                std::vector<std::string> strs = CL::utility::split(line);
                size_t NIrreds = (strs.size() - 1) / 2;
                loader.CNPI2point.resize(NIrreds);
                for (size_t i = 0; i < NIrreds; i++) {
                    size_t CNPI  = std::stoul(strs[i]) - 1;
                    size_t point = std::stoul(strs[NIrreds + 1 + i]) - 1;
                    loader.CNPI2point[CNPI] = point;
                }
            }
            ifs.close();
        }
        template <typename T> void load_pointDefs(std::vector<T> & loaders, const std::string & data_directory) const {
            // count number of point group irreducibles
            size_t NIrreds;
            std::string file_pattern = data_directory + "irred*.IntCoordDef";
            glob_t gl;
            if(glob(file_pattern.c_str(), GLOB_NOSORT, NULL, & gl) == 0) NIrreds = gl.gl_pathc;
            else throw CL::utility::file_error(file_pattern);
            globfree(& gl);
            // fill in loader.point_defs
            for (auto & loader : loaders) {
                loader.point_defs.resize(NIrreds);
                for (size_t i = 0; i < NIrreds; i++) loader.point_defs[i] = "irred" + std::to_string(i + 1) + ".IntCoordDef";
            }
        }

        // Read geometries in symmetry adapted internal coordinates
        std::shared_ptr<DataSet<SAGeometry>> read_SAGeomSet() const;
        // Read Hamiltonians in symmetry adapted internal coordinates
        std::tuple<std::shared_ptr<DataSet<RegSAHam>>, std::shared_ptr<DataSet<DegSAHam>>> read_SAHamSet() const;
};

} // namespace abinitio

#endif