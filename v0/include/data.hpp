#ifndef data_hpp
#define data_hpp

#include <abinitio/SAHamiltonian.hpp>
#include <abinitio/DataSet.hpp>

class RegHam : public abinitio::RegSAHam {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
    public:
        RegHam();
        RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &));
        ~RegHam();

        CL::utility::matrix<at::Tensor> xs() const;
        CL::utility::matrix<at::Tensor> JxqTs() const;
};

class DegHam : public abinitio::DegSAHam {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
    public:
        DegHam();
        DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
               std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &));
        ~DegHam();

        CL::utility::matrix<at::Tensor> xs() const;
        CL::utility::matrix<at::Tensor> JxqTs() const;
};

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list, const double & zero_point, const double & weight);

#endif