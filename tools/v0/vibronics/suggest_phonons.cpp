#include <Foptim/Foptim.hpp>

#include <tchem/chemistry.hpp>

namespace {
    // constraint (T . Q + b)^T . sigmainv_ . (T . Q + b) - contour^2 = 0
    double contour_;
    at::Tensor T_, b_, sigmainv_;
    // Control minimizing or maximizing merit function Q[index]
    // i.e. the merit function is modified to sign_ * Q[index]
    // if sign_ > 0: minimizing Q[index] by minimizing sign_ * Q[index]
    // else        : maximizing Q[index] by minimizing sign_ * Q[index]
    double sign_;
    size_t index_;

    void f(double & fQ, const double * Q, const int32_t & intdim) {
        fQ = sign_ * Q[index_];
    }
    void f_fd(double & fQ, double * fdQ, const double * Q, const int32_t & intdim) {
        fQ = sign_ * Q[index_];
        std::memset(fdQ, 0.0, intdim * sizeof(double));
        fdQ[index_] = sign_;
    }
    void fdd(double * fddQ, const double * Q, const int32_t & intdim) {
        std::memset(fddQ, 0.0, intdim * intdim * sizeof(double));
    }

    void c(double * cQ, const double * Q, const int32_t & Nc, const int32_t & intdim) {
        at::Tensor Q_tensor = at::from_blob(const_cast<double *>(Q), intdim, c10::TensorOptions().dtype(torch::kFloat64));
        at::Tensor q = T_.mv(Q_tensor) + b_;
        cQ[0] = (q.dot(sigmainv_.mv(q))).item<double>() - contour_ * contour_;
    }
    void c_cd(double * cQ, double * cdQ, const double * Q, const int32_t & Nc, const int32_t & intdim) {
        at::Tensor Q_tensor = at::from_blob(const_cast<double *>(Q), intdim, c10::TensorOptions().dtype(torch::kFloat64));
        at::Tensor q = T_.mv(Q_tensor) + b_;
        // c
        cQ[0] = (q.dot(sigmainv_.mv(q))).item<double>() - contour_ * contour_;
        // cd
        at::Tensor cdQ_tensor = at::from_blob(cdQ, intdim, c10::TensorOptions().dtype(torch::kFloat64));
        cdQ_tensor.copy_(2.0 * T_.transpose(0, 1).mv(sigmainv_.mv(q)));
    }
    void c_cd_cdd(double * cQ, double * cdQ, double * cddQ, const double * Q, const int32_t & Nc, const int32_t & intdim) {
        at::Tensor Q_tensor = at::from_blob(const_cast<double *>(Q), intdim, c10::TensorOptions().dtype(torch::kFloat64));
        at::Tensor q = T_.mv(Q_tensor) + b_;
        // c
        cQ[0] = (q.dot(sigmainv_.mv(q))).item<double>() - contour_ * contour_;
        // cd
        at::Tensor cdQ_tensor = at::from_blob(cdQ, intdim, c10::TensorOptions().dtype(torch::kFloat64));
        cdQ_tensor.copy_(2.0 * T_.transpose(0, 1).mv(sigmainv_.mv(q)));
        // cdd
        at::Tensor cddQ_tensor = at::from_blob(cddQ, {intdim, intdim}, c10::TensorOptions().dtype(torch::kFloat64));
        cddQ_tensor.copy_(2.0 * T_.transpose(0, 1).mm(sigmainv_.mm(T_)));
    }
}

void suggest_phonons(const double & contour,
const std::vector<size_t> & init_NModes, const std::vector<size_t> & final_NModes,
const at::Tensor & init_q, const at::Tensor & final_q,
const tchem::chem::SANormalMode & init_vib, const tchem::chem::SANormalMode & final_vib) {
    int64_t intdim = init_q.size(0);
    // Set contour, T, b, sigma^-1
    contour_ = contour;
    at::Tensor init_Linv = init_q.new_zeros({intdim, intdim});
    size_t start = 0;
    for (size_t i = 0; i < init_NModes.size(); i++) {
        size_t stop = start + init_NModes[i];
        init_Linv.slice(0, start, stop).slice(1, start, stop).copy_(init_vib.Linvs()[i]);
        start = stop;
    }
    at::Tensor final_L = final_q.new_zeros({intdim, intdim});
    start = 0;
    for (size_t i = 0; i < final_NModes.size(); i++) {
        size_t stop = start + final_NModes[i];
        final_L.slice(0, start, stop).slice(1, start, stop).copy_(final_vib.intmodes()[i]);
        start = stop;
    }
    final_L.transpose_(0, 1);
    T_ = init_Linv.mm(final_L);
    b_ = init_Linv.mv(final_q - init_q);
    sigmainv_ = at::diag(2.0 * at::cat(init_vib.frequencies()));
    // Get lower and upper bounds
    std::vector<double> lower_bound(intdim), upper_bound(intdim);
    for (size_t i = 0; i < intdim; i++) {
        // lower bound
        at::Tensor Q = init_q.new_zeros(intdim);
        sign_ = 1.0;
        index_ = i;
        Foptim::ALagrangian_Newton_Raphson(f, f_fd, fdd, c, c_cd, c_cd_cdd, Q.data_ptr<double>(), intdim, 1);
        lower_bound[i] = Q[i].item<double>();
        // upper bound
        Q.fill_(0.0);
        sign_ = -1.0;
        Foptim::ALagrangian_Newton_Raphson(f, f_fd, fdd, c, c_cd, c_cd_cdd, Q.data_ptr<double>(), intdim, 1);
        upper_bound[i] = Q[i].item<double>();
    }
    // Map the continuum bounds to discrete phonons
    std::vector<double> dble_phonons(intdim);
    std::vector<size_t> phonons(intdim);
    at::Tensor final_freqs = at::cat(final_vib.frequencies());
    for (size_t i = 0; i < intdim; i++) {
        double bound = std::max(std::abs(lower_bound[i]), std::abs(upper_bound[i]));
        dble_phonons[i] = final_freqs[i].item<double>() * bound * bound - 0.5;
        phonons[i] = ceil(dble_phonons[i]);
    }
    // Output
    std::ofstream ofs;
    std::cout << "Please refer to continuum.txt and phonons.txt for vibrational basis phonon suggestions\n";
    ofs.open("continuum.txt"); {
        ofs << "Contour value = " << contour << '\n';
        size_t NVib = 1;
        for (size_t i = 0; i < init_NModes[0]; i++) NVib *= (phonons[i] + 1);
        ofs << NVib << " = number of vibrational basis functions from initial-state totally symemtric irreducible\n";
        for (size_t i = init_NModes[0]; i < intdim; i++) NVib *= (phonons[i] + 1);
        ofs << NVib << " = number of vibrational basis functions\n";
        ofs << "mode    lower bound    upper bound    phonons    continuum phonons\n";
        for (size_t i = 0; i < intdim; i++)
        ofs << std::setw(4) << i + 1 << "    "
            << std::fixed << std::setw(11) << std::setprecision(5) << lower_bound[i] << "    "
            << std::fixed << std::setw(11) << std::setprecision(5) << upper_bound[i] << "    "
            << std::setw(7) << phonons[i] << "    "
            << std::fixed << std::setw(17) << std::setprecision(5) << dble_phonons[i] << '\n';
    }
    ofs.close();
    ofs.open("phonons.txt"); {
        ofs << "phonons for initial-state totally symemtric irreducible only:\n";
        ofs << "(";
        for (size_t i = 0; i < init_NModes[0] - 1; i++) ofs << phonons[i] << ", ";
        ofs << phonons[init_NModes[0] - 1] << "),\n";
        size_t count = init_NModes[0];
        for (size_t irred = 1; irred < init_NModes.size(); irred++) {
            ofs << "(";
            for (size_t i = 0; i < init_NModes[irred] - 1; i++) {
                ofs << 0 << ", ";
                count++;
            }
            ofs << 0 << "),\n";
            count++;
        }
        ofs << "phonons:\n";
        count = 0;
        for (size_t irred = 0; irred < init_NModes.size(); irred++) {
            ofs << "(";
            for (size_t i = 0; i < init_NModes[irred] - 1; i++) {
                ofs << phonons[count] << ", ";
                count++;
            }
            ofs << phonons[count] << "),\n";
            count++;
        }
    }
    ofs.close();
}