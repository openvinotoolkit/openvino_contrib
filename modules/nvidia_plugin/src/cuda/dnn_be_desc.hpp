// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gsl/span>

#include "dnn_be_attrs.hpp"
#include "runtime.hpp"

namespace CUDA {

/**
 * @brief cuDNN Backend Descriptor wrapper.
 *
 * Every meaningful entity in cuDNN backend API is an opaque backend descriptor
 * object. Each descriptor has a type (cudnnBackendDescriptorType_t) and a set
 * of attributes (cudnnBackendAttributeName_t).
 */
class DnnBackendDescriptor : public Handle<cudnnBackendDescriptor_t> {
public:
    template <cudnnBackendAttributeName_t Name>
    using ValueType = typename DnnBEAttrType<GetDnnBEAttrTypeId<Name>()>::ValueType;

    DnnBackendDescriptor(cudnnBackendDescriptorType_t descriptorType)
        : Handle{cudnnBackendCreateDescriptorAdapter, cudnnBackendDestroyDescriptor, descriptorType} {}

protected:
    static cudnnStatus_t cudnnBackendCreateDescriptorAdapter(cudnnBackendDescriptor_t* descriptor,
                                                             cudnnBackendDescriptorType_t descriptorType) {
        return cudnnBackendCreateDescriptor(descriptorType, descriptor);
    }

    template <cudnnBackendAttributeName_t Name>
    int64_t getAttributeValueCount() const {
        int64_t num_values = 0;
        throwIfError(::cudnnBackendGetAttribute(get(), Name, GetDnnBEAttrTypeId<Name>(), 0, &num_values, nullptr));
        return num_values;
    }

    template <cudnnBackendAttributeName_t Name>
    std::vector<ValueType<Name>> getAttributeValues() const {
        std::vector<ValueType<Name>> values(getAttributeValueCount<Name>());
        getAttributeValues<Name>(values);
        return values;
    }

    template <cudnnBackendAttributeName_t Name,
              class T,
              typename = typename std::enable_if<std::is_convertible<T*, DnnBackendDescriptor*>::value>::type>
    std::vector<std::shared_ptr<T>> getBEDescAttributeValues() const {
        std::vector<std::shared_ptr<T>> values(getAttributeValueCount<Name>());
        for (auto& val : values) {
            val = std::make_shared<T>();
        }
        std::vector<cudnnBackendDescriptor_t> raw_be_descs;
        std::transform(
            values.begin(), values.end(), std::back_inserter(raw_be_descs), [](const auto& val) { return val->get(); });
        getAttributeValues<Name>(raw_be_descs);
        values.resize(raw_be_descs.size());
        return values;
    }

    template <cudnnBackendAttributeName_t Name>
    ValueType<Name> getAttributeValue() const {
        int64_t num_values = 0;
        ValueType<Name> value{};
        throwIfError(::cudnnBackendGetAttribute(get(), Name, GetDnnBEAttrTypeId<Name>(), 1, &num_values, &value));
        Ensures(1 == num_values);
        return value;
    }

private:
    template <cudnnBackendAttributeName_t Name>
    void getAttributeValues(std::vector<ValueType<Name>>& io_values) const {
        int64_t num_values = -1;
        throwIfError(::cudnnBackendGetAttribute(
            get(), Name, GetDnnBEAttrTypeId<Name>(), io_values.size(), &num_values, io_values.data()));
        {
            // NOTE: Implementing workaround for cuDNN v8.1 bug, when sometimes the number of actually
            //       returned attributes is smaller than previously returned by `getAttributeValueCount()`.
            if (io_values.size() > num_values) {
                io_values.resize(num_values);
            }
        }
        Ensures(io_values.size() == num_values);
    }
};

/**
 * @brief cuDNN Backend Descriptor builder.
 *
 * This class creates backend descriptor. Also, it provides a wrapper
 * methods to set its attributes.
 */
template <typename T>
class DnnBackendDescriptorBuilder {
public:
    template <cudnnBackendAttributeName_t Name>
    using ValueType = typename DnnBEAttrType<GetDnnBEAttrTypeId<Name>()>::ValueType;

    DnnBackendDescriptorBuilder() {}
    virtual ~DnnBackendDescriptorBuilder() = 0;

    virtual std::shared_ptr<T> build() {
        throwIfError(::cudnnBackendFinalize(desc_->get()));
        return desc_;
    }

protected:
    template <cudnnBackendAttributeType_t TypeID>
    void setAttributeValues(cudnnBackendAttributeName_t name,
                            gsl::span<const typename DnnBEAttrType<TypeID>::ValueType> values) {
        throwIfError(::cudnnBackendSetAttribute(desc_->get(), name, TypeID, values.size(), values.data()));
    }

    template <cudnnBackendAttributeName_t Name>
    void setAttributeValues(gsl::span<const ValueType<Name>> values) {
        setAttributeValues<GetDnnBEAttrTypeId<Name>()>(Name, values);
    }

    template <cudnnBackendAttributeType_t TypeID>
    void setAttributeValue(cudnnBackendAttributeName_t name, typename DnnBEAttrType<TypeID>::ValueType value) {
        setAttributeValues<TypeID>(name, gsl::span<const typename DnnBEAttrType<TypeID>::ValueType>{&value, 1});
    }

    template <cudnnBackendAttributeName_t Name>
    void setAttributeValue(ValueType<Name> value) {
        setAttributeValue<GetDnnBEAttrTypeId<Name>()>(Name, value);
    }

    std::shared_ptr<T> desc_ = std::make_shared<T>();
};

template <typename T>
inline DnnBackendDescriptorBuilder<T>::~DnnBackendDescriptorBuilder() {}

}  // namespace CUDA
