#include "../include/global.hpp"

// read parameters from prefix_state1-state2_layer.txt into x
void read_parameters(const std::string & prefix, at::Tensor & x) {
    auto pmat = Hdnet->parameters();
    size_t start = 0;
    for (size_t istate = 0     ; istate < Hdnet->NStates(); istate++)
    for (size_t jstate = istate; jstate < Hdnet->NStates(); jstate++) {
        std::string prefix_now = prefix + "_" + std::to_string(istate + 1) + "-" + std::to_string(jstate + 1) + "_";
        const auto & ps = pmat[istate][jstate];
        // The 1st layer is interpretable
        std::string file = prefix_now + "1.txt";
        std::ifstream ifs; ifs.open(file);
        at::Tensor A = ps[0].new_empty(ps[0].sizes());
        for (size_t i = 0; i < A.size(1); i++) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != A.size(0)) throw std::invalid_argument("inconsisten line");
            for (size_t j = 0; j < A.size(0); j++) A[j][i].fill_(std::stod(strs[j]));
        }
        size_t stop = start + A.numel();
        x.slice(0, start, stop).copy_(A.view(A.numel()));
        start = stop;
        if (Hdnet->irreds()[istate][jstate] == 0) {
            at::Tensor b = ps[1].new_empty(ps[1].sizes());
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            std::getline(ifs, line);
            if (! ifs.good()) throw CL::utility::file_error(file);
            auto strs = CL::utility::split(line);
            if (strs.size() != b.size(0)) throw std::invalid_argument("inconsisten line");
            for (size_t j = 0; j < A.size(0); j++) b[j].fill_(std::stod(strs[j]));
            size_t stop = start + b.numel();
            x.slice(0, start, stop).copy_(b);
            start = stop;
        }
        ifs.close();
        // The other layers are uninterpretable
        if (Hdnet->irreds()[istate][jstate] == 0)
        for (size_t layer = 2; layer < ps.size(); layer += 2) {
            std::string file = prefix_now + std::to_string(layer / 2 + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            at::Tensor A = ps[layer].new_empty(ps[layer].sizes());
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            at::Tensor b = ps[layer + 1].new_empty(ps[layer + 1].sizes());
            ifs >> line;
            for (size_t i = 0; i < b.size(0); i++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                b[i].fill_(dbletemp);
            }
            size_t stop = start + A.numel();
            x.slice(0, start, stop).copy_(A.view(A.numel()));
            start = stop;
            stop = start + b.numel();
            x.slice(0, start, stop).copy_(b);
            start = stop;
            ifs.close();
        }
        else
        for (size_t layer = 1; layer < ps.size(); layer++) {
            std::string file = prefix_now + std::to_string(layer + 1) + ".txt";
            std::ifstream ifs; ifs.open(file);
            at::Tensor A = ps[layer].new_empty(ps[layer].sizes());
            std::string line;
            std::getline(ifs, line);
            for (size_t i = 0; i < A.size(1); i++)
            for (size_t j = 0; j < A.size(0); j++) {
                double dbletemp;
                ifs >> dbletemp;
                if (! ifs.good()) throw CL::utility::file_error(file);
                A[j][i].fill_(dbletemp);
            }
            size_t stop = start + A.numel();
            x.slice(0, start, stop).copy_(A.view(A.numel()));
            start = stop;
            ifs.close();
        }
    }
}
