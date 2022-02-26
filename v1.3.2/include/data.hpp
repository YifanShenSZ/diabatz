#ifndef data_hpp
#define data_hpp

#include <abinitio/DataSet.hpp>

#include "data_classes.hpp"

std::tuple<std::shared_ptr<abinitio::DataSet<RegHam>>, std::shared_ptr<abinitio::DataSet<DegHam>>>
read_data(const std::vector<std::string> & user_list, const double & zero_point);

std::shared_ptr<abinitio::DataSet<Energy>> read_energy(const std::vector<std::string> & user_list, const double & zero_point);

// given a data set,
// return the average and the standard deviation of the input layers
template <typename T>
std::tuple<CL::utility::matrix<at::Tensor>, CL::utility::matrix<at::Tensor>>
statisticize_input(const std::shared_ptr<abinitio::DataSet<T>> & set) {
    size_t NExamples = set->size_int();
    if (NExamples == 0) return std::make_tuple(CL::utility::matrix<at::Tensor>(0), CL::utility::matrix<at::Tensor>(0));
    // 0th example
    const auto & example = set->examples()[0];
    const auto & xs = example->xs();
    CL::utility::matrix<at::Tensor>  x_sum = xs,
                                    xx_sum = xs * xs;
    // remaining examples
    for (size_t i = 1; i < NExamples; i++) {
        const auto & example = set->examples()[i];
        const auto & xs = example->xs();
         x_sum += xs;
        xx_sum += xs * xs;
    }
    // compute average
    CL::utility::matrix<at::Tensor> avg =  x_sum / at::tensor((double)NExamples);
    // For asymmetric irreducibles, the average is always 0 because
    // the network design has considered the symmetry,
    // which effectively has performed a data augmentation that considers all symmetry-equivalent geometries,
    // and the asymmetric average of all those geometires is always 0
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    if (Hdnet->irreds()[i][j] != 0) avg[i][j].fill_(0.0);
    // compute standard deviation
    CL::utility::matrix<at::Tensor> std = xx_sum / at::tensor((double)NExamples) - avg * avg;
    for (size_t i = 0; i < Hdnet->NStates(); i++)
    for (size_t j = i; j < Hdnet->NStates(); j++)
    std[i][j].sqrt_();
    return std::make_tuple(avg, std);
}

#endif