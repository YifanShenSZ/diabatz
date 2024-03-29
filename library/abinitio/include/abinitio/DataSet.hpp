#ifndef abinitio_DataSet_hpp
#define abinitio_DataSet_hpp

#include <torch/torch.h>

namespace abinitio {

// A class to hold data set & to make data loader from
// T must have method `to`, e.g. at::Tensor
template <class T> class DataSet : public torch::data::datasets::Dataset<DataSet<T>, std::shared_ptr<T>> {
    private:
        std::vector<std::shared_ptr<T>> examples_;
    public:
        inline DataSet(const std::vector<std::shared_ptr<T>> & _examples) : examples_(_examples) {}
        inline ~DataSet() {}

        // override the size method to infer the size of the data set
        inline torch::optional<size_t> size() const override {return examples_.size();}
        // override the get method to load custom data
        inline std::shared_ptr<T> get(size_t index) override {return examples_[index];}

        inline const std::vector<std::shared_ptr<T>> & examples() const {return examples_;}

        inline size_t size_int() const {return examples_.size();}

        inline void to(const c10::DeviceType & device) {
            for (const auto & data : examples_) data->to(device);
        }        
};

} // namespace abinitio

#endif