#ifndef data_classes_hpp
#define data_classes_hpp

#include <abinitio/SAenergy.hpp>
#include <abinitio/SAHamiltonian.hpp>

class Energy : public abinitio::SAEnergy {
    private:
        // input layers and their transposed Jacobians over Cartesian coordinate
        CL::utility::matrix<at::Tensor> x1s_, Jx1rTs_, x2s_, Jx2rTs_;
    public:
        Energy();
        Energy(const std::shared_ptr<abinitio::SAEnergy> & ener,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &)
        );
        ~Energy();

        const CL::utility::matrix<at::Tensor> & x1s() const;
        const CL::utility::matrix<at::Tensor> & Jx1rTs() const;
        const CL::utility::matrix<at::Tensor> & x2s() const;
        const CL::utility::matrix<at::Tensor> & Jx2rTs() const;

        void scale_features(
            const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
            const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2
        );
};

class RegHam : public abinitio::RegSAHam {
    private:
        // input layers and their transposed Jacobians over Cartesian coordinate
        CL::utility::matrix<at::Tensor> x1s_, Jx1rTs_, x2s_, Jx2rTs_;
        // the pretrained part of Hd and ▽Hd
        at::Tensor pretrained_Hd_, pretrained_DrHd_;
    public:
        RegHam();
        RegHam(const std::shared_ptr<abinitio::RegSAHam> & ham,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &)
        );
        ~RegHam();

        const CL::utility::matrix<at::Tensor> & x1s() const;
        const CL::utility::matrix<at::Tensor> & Jx1rTs() const;
        const CL::utility::matrix<at::Tensor> & x2s() const;
        const CL::utility::matrix<at::Tensor> & Jx2rTs() const;

        void scale_features(
            const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
            const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2
        );
};

class DegHam : public abinitio::DegSAHam {
    private:
        // input layers and their transposed Jacobians over Cartesian coordinate
        CL::utility::matrix<at::Tensor> x1s_, Jx1rTs_, x2s_, Jx2rTs_;
        // the pretrained part of Hd and ▽Hd
        at::Tensor pretrained_Hd_, pretrained_DrHd_;
    public:
        DegHam();
        DegHam(const std::shared_ptr<abinitio::DegSAHam> & ham,
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x1)(const std::vector<at::Tensor> &),
            std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>> (*q2x2)(const std::vector<at::Tensor> &)
        );
        ~DegHam();

        const CL::utility::matrix<at::Tensor> & x1s() const;
        const CL::utility::matrix<at::Tensor> & Jx1rTs() const;
        const CL::utility::matrix<at::Tensor> & x2s() const;
        const CL::utility::matrix<at::Tensor> & Jx2rTs() const;

        void scale_features(
            const CL::utility::matrix<at::Tensor> & shift1, const CL::utility::matrix<at::Tensor> & width1,
            const CL::utility::matrix<at::Tensor> & shift2, const CL::utility::matrix<at::Tensor> & width2
        );
};

#endif