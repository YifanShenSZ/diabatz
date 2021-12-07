#ifndef data_classes_hpp
#define data_classes_hpp

#include <abinitio/SAenergy.hpp>
#include <abinitio/SAHamiltonian.hpp>

class Energy : public abinitio::SAEnergy {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
    public:
        Energy();
        Energy(const std::shared_ptr<abinitio::SAEnergy> & ener,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &));
        ~Energy();

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
};

class RegHam : public abinitio::RegSAHam {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
    public:
        RegHam();
        RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &));
        ~RegHam();

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
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

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
};

#endif