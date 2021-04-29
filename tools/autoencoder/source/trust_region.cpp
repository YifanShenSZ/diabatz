#include <omp.h>

#include <Foptim/trust_region.hpp>

#include "../include/global.hpp"

namespace {
    int32_t NEqs, NPars;

    size_t OMP_NUM_THREADS;
    // Each thread owns a copy of the autoencoder network
    // Thread 0 shares the original autoencoder network
    std::vector<std::shared_ptr<DimRed::Encoder>> encoders;
    std::vector<std::shared_ptr<DimRed::Decoder>> decoders;
    // Each thread owns a chunk of data
    std::vector<std::vector<std::shared_ptr<abinitio::SAGeometry>>> chunk;
    // Each thread works on a segment of residue or Jacobian
    // Thread i works on rows [segstart[i], segstart[i + 1])
    std::vector<size_t> segstart;

    void initialize() {
        NEqs = geom_set->size_int() * geom_set->get(0)->qs()[irreducible].size(0);
        std::cout << "The data set corresponds to " << NEqs << " least sqaure equations\n";
        NPars = 0;
        for (const auto & p : encoder->parameters()) NPars += p.numel();
        for (const auto & p : decoder->parameters()) NPars += p.numel();
        std::cout << "There are " << NPars << " parameters to train\n\n";

        OMP_NUM_THREADS = omp_get_max_threads();
        std::cout << "The number of threads = " << OMP_NUM_THREADS << '\n';

        encoders.resize(OMP_NUM_THREADS);
        decoders.resize(OMP_NUM_THREADS);
        encoders[0] = encoder;
        decoders[0] = decoder;
        for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
            encoders[i] = std::make_shared<DimRed::Encoder>(encoder);
            decoders[i] = std::make_shared<DimRed::Decoder>(decoder);
            encoders[i]->train();
            decoders[i]->train();
        }

        chunk.resize(OMP_NUM_THREADS);
        size_t chunk_size = geom_set->size_int() / OMP_NUM_THREADS;
        std::cout << "Each thread owns " << chunk_size << " data points\n";
        size_t count = 0;
        // Thread 0 to OMP_NUM_THREADS - 2 each owns `chunk_size` data points
        for (size_t thread = 0; thread < OMP_NUM_THREADS - 1; thread++) {
            chunk[thread].resize(chunk_size);
            for (auto & data : chunk[thread]) {
                data = geom_set->get(count);
                count++;
            }
        }
        // The last thread owns all remaining data points
        chunk.back().resize(geom_set->size_int() - (OMP_NUM_THREADS - 1) * chunk_size);
        for (auto & data : chunk.back()) {
            data = geom_set->get(count);
            count++;
        }

        segstart.resize(OMP_NUM_THREADS);
        segstart[0] = 0;
        for (size_t thread = 1; thread < OMP_NUM_THREADS; thread++)
        segstart[thread] = segstart[thread - 1]
                         + chunk[thread - 1].size()
                         * geom_set->get(0)->qs()[irreducible].size(0);
        for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++)
        std::cout << "Thread " << thread << " starts with Jacobian row " << segstart[thread] << '\n';
    }

    inline void p2c(const size_t & thread, double * c) {
        size_t count = 0;
        for (const at::Tensor & p : encoders[thread]->parameters()) {
            size_t numel = p.numel();
            std::memcpy(&(c[count]), p.data_ptr<double>(), numel * sizeof(double));
            count += numel;
        }
        for (const at::Tensor & p : decoders[thread]->parameters()) {
            size_t numel = p.numel();
            std::memcpy(&(c[count]), p.data_ptr<double>(), numel * sizeof(double));
            count += numel;
        }
    }
    inline void c2p(const double * c, const size_t & thread) {
        size_t count = 0;
        for (const at::Tensor & p : encoders[thread]->parameters()) {
            size_t numel = p.numel();
            std::memcpy(p.data_ptr<double>(), &(c[count]), numel * sizeof(double));
            count += numel;
        }
        for (const at::Tensor & p : decoders[thread]->parameters()) {
            size_t numel = p.numel();
            std::memcpy(p.data_ptr<double>(), &(c[count]), numel * sizeof(double));
            count += numel;
        }
    }

    void residue(double * r, const double * c, const int32_t & M, const int32_t & N) {
        #pragma omp parallel for
        for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            size_t start = segstart[thread];
            for (const auto & data : chunk[thread]) {
                const auto & x = data->qs()[irreducible];
                at::Tensor y = decoders[thread]->forward(encoders[thread]->forward(x))
                             - x;
                size_t numel = y.numel();
                std::memcpy(&(r[start]), y.data_ptr<double>(), numel * sizeof(double));
                start += numel;
            }
        }
    }

    void Jacobian(double * JT, const double * c, const int32_t & M, const int32_t & N) {
        at::Tensor J = at::from_blob(JT, {N, M}, at::TensorOptions().dtype(torch::kFloat64));
        J.transpose_(0, 1);
        #pragma omp parallel for
        for (size_t thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, thread);
            auto parameters = encoders[thread]->parameters(),
                 depars     = decoders[thread]->parameters();
            parameters.insert(parameters.end(), depars.begin(), depars.end());
            size_t start = segstart[thread];
            for (const auto & data : chunk[thread]) {
                at::Tensor y = decoders[thread]->forward(encoders[thread]->forward(data->qs()[irreducible]));
                for (size_t i = 0; i < y.numel(); i++) {
                    auto gs = torch::autograd::grad({y[i]}, parameters, {}, true);
                    for (at::Tensor & g : gs) g = g.view(g.numel());
                    at::Tensor g = at::cat(gs);
                    J[start].copy_(g);
                    start++;
                }
            }
        }
    }
}

void trust_region(const size_t & max_iteration) {
    initialize();
    std::cout << '\n';
    double * c = new double[NPars];
    p2c(0, c);
    std::cout << "The initial RMSD = " << RMSD() << std::endl;
    Foptim::trust_region(residue, Jacobian, c, NEqs, NPars, max_iteration);
    c2p(c, 0);
    std::cout << "The final RMSD = " << RMSD() << std::endl;
    delete [] c;
}