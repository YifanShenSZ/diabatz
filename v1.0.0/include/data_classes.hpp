#ifndef data_classes_hpp
#define data_classes_hpp

#include <abinitio/SAenergy.hpp>
#include <abinitio/SAHamiltonian.hpp>

#include <Hd/kernel.hpp>

class Energy : public abinitio::SAEnergy {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
        // the pretrained part of Hd
        at::Tensor pretrained_Hd_;
    public:
        Energy();
        Energy(const std::shared_ptr<abinitio::SAEnergy> & ener,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
            const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel);
        ~Energy();

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
        const at::Tensor & pretrained_Hd() const;
};

class RegHam : public abinitio::RegSAHam {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
        // the pretrained part of Hd and ▽Hd
        at::Tensor pretrained_Hd_, pretrained_DqHd_;
    public:
        RegHam();
        RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
            const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel);
        ~RegHam();

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
        const at::Tensor & pretrained_Hd  () const;
        const at::Tensor & pretrained_DqHd() const;
};

class DegHam : public abinitio::DegSAHam {
    private:
        // input layers and their transposed Jacobians over internal coordinate
        CL::utility::matrix<at::Tensor> xs_, JxqTs_;
        // the pretrained part of Hd and ▽Hd
        at::Tensor pretrained_Hd_, pretrained_DqHd_;
    public:
        DegHam();
        DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
               std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x)(const std::vector<at::Tensor> &),
               const std::shared_ptr<Hd::kernel> & pretrained_Hdkernel);
        ~DegHam();

        const CL::utility::matrix<at::Tensor> & xs() const;
        const CL::utility::matrix<at::Tensor> & JxqTs() const;
        const at::Tensor & pretrained_Hd  () const;
        const at::Tensor & pretrained_DqHd() const;
};

#endif