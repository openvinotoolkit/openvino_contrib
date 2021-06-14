// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>
#include "runtime.hpp"
#include "dnn_be_attrs.hpp"

namespace CUDA {

/**
 * @brief cuDNN Backend Descriptor wrapper.
 *
 * Every meaningful entity in cuDNN backend API is an opaque backend descriptor
 * object. Each descriptor has a type (cudnnBackendDescriptorType_t) and a set
 * of attributes (cudnnBackendAttributeName_t).
 *
 * This class creates and owns backend descriptor. Also, it provides a wrapper
 * methods to manage its attributes.
 */
class DnnBackendDescriptor {
public:
    DnnBackendDescriptor(cudnnBackendDescriptorType_t descriptorType)
        : owner_ {create(descriptorType)} {
    }

    cudnnBackendDescriptor_t get() const { return owner_.get(); }

    template<cudnnBackendAttributeName_t Name>
    using ValueType = typename DnnBEAttrTypeID<DnnBEAttrName<Name>::TypeID>::ValueType;

    template<cudnnBackendAttributeType_t TypeID>
    void setAttributeValues(cudnnBackendAttributeName_t name,
                            gsl::span<typename DnnBEAttrTypeID<TypeID>::ValueType> values) {
        throwIfError(::cudnnBackendSetAttribute(get(), name, TypeID, values.size(), values.data()));
    }

    template<cudnnBackendAttributeName_t Name>
    void setAttributeValues(gsl::span<ValueType<Name>> values) {
        setAttributeValues<DnnBEAttrName<Name>::TypeID>(Name, values);
    }

    template<cudnnBackendAttributeType_t TypeID>
    void setAttributeValue(cudnnBackendAttributeName_t name,
                           typename DnnBEAttrTypeID<TypeID>::ValueType value) {
        setAttributeValues<TypeID>(name, gsl::span<typename DnnBEAttrTypeID<TypeID>::ValueType> {&value, 1});
    }

    template<cudnnBackendAttributeName_t Name>
    void setAttributeValue(ValueType<Name> value) {
        setAttributeValue<DnnBEAttrName<Name>::TypeID>(Name, value);
    }

    void finalize() {
        throwIfError(::cudnnBackendFinalize(get()));
    }

    template<cudnnBackendAttributeName_t Name>
    int64_t getAttributeValueCount() const {
        int64_t num_values = 0;
        throwIfError(::cudnnBackendGetAttribute(get(), Name,  DnnBEAttrName<Name>::TypeID,
                                                0, &num_values, nullptr));
        return num_values;
    }

    template<cudnnBackendAttributeName_t Name>
    std::vector<ValueType<Name>> getAttributeValues() const {
        std::vector<ValueType<Name>> values(getAttributeValueCount<Name>());
        getAttributeValues<Name>(values);
        return values;
    }

    template<cudnnBackendAttributeName_t Name, class T,
             typename = std::enable_if_t<std::is_convertible_v<T*, DnnBackendDescriptor*>>>
    std::vector<T> getBEDescAttributeValues() const {
        std::vector<T> values(getAttributeValueCount<Name>());
        std::vector<cudnnBackendDescriptor_t> raw_be_descs;
        std::transform(values.begin(), values.end(), std::back_inserter(raw_be_descs),
                       std::bind(&T::get, std::placeholders::_1));
        getAttributeValues<Name>(raw_be_descs);
        values.resize(raw_be_descs.size());
        return values;
    }

    template<cudnnBackendAttributeName_t Name>
    ValueType<Name> getAttributeValue() const {
        int64_t num_values = 0;
        ValueType<Name> value {};
        throwIfError(::cudnnBackendGetAttribute(get(), Name, DnnBEAttrName<Name>::TypeID,
                                                1, &num_values, &value));
        Ensures(1 == num_values);
        return value;
    }

private:
    template<cudnnBackendAttributeName_t Name>
    void getAttributeValues(std::vector<ValueType<Name>>& io_values) const {
        int64_t num_values = -1;
        throwIfError(::cudnnBackendGetAttribute(get(), Name, DnnBEAttrName<Name>::TypeID,
                                                io_values.size(), &num_values, io_values.data()));
        {
            // NOTE: Implementing workaround for cuDNN v8.1 bug, when sometimes the number of actually
            //       returned attributes is smaller than previously returned by `getAttributeValueCount()`.
            if (io_values.size() > num_values) {
                io_values.resize(num_values);
            }
        }
        Ensures(io_values.size() == num_values);
    }

private:
    static cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t descriptorType) {
        cudnnBackendDescriptor_t desc {};
        throwIfError(::cudnnBackendCreateDescriptor(descriptorType, &desc));
        return desc;
    }
    struct Deleter {
        void operator()(cudnnBackendDescriptor_t desc) const noexcept {
            logIfError(::cudnnBackendDestroyDescriptor(desc));
        }
    };
    std::unique_ptr<std::remove_pointer_t<cudnnBackendDescriptor_t>, Deleter> owner_;
};

} // namespace CUDA
