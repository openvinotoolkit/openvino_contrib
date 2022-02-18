// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cuda_runtime_api.h>

#include <functional>
#include <cuda/device_pointers.hpp>
#include <error.hpp>
#include <atomic>

#include "props.hpp"

inline void throwIfError(
    cudaError_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != cudaSuccess) CUDAPlugin::throwIEException(cudaGetErrorString(err), location);
}

inline void logIfError(
    cudaError_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != cudaSuccess) CUDAPlugin::logError(cudaGetErrorString(err), location);
}

namespace CUDA {

template <typename T>
auto toNative(T&& a) noexcept(noexcept(std::forward<T>(a).get())) -> decltype(std::forward<T>(a).get()) {
    return std::forward<T>(a).get();
}

template <typename T>
auto toNative(T&& a) noexcept(noexcept(std::forward<T>(a).data())) -> decltype(std::forward<T>(a).data()) {
    return std::forward<T>(a).data();
}

template <typename T>
typename std::enable_if<std::is_scalar<typename std::decay<T>::type>::value, typename std::decay<T>::type>::type
toNative(T t) noexcept {
    return t;
}

template <typename T, typename R, typename... Args>
auto createFirstArg(R (*creator)(T*, Args... args), Args... args) {
    T t;
    throwIfError(creator(&t, toNative(std::forward<Args>(args))...));
    return t;
}

template <typename R, typename... ConArgs, typename... Args>
auto createLastArg(R (*creator)(ConArgs...), Args... args) {
    using LastType = typename std::remove_pointer<
        typename std::tuple_element<sizeof...(ConArgs) - 1, std::tuple<ConArgs...>>::type>::type;
    LastType t;
    throwIfError(creator(toNative(std::forward<Args>(args))..., &t));
    return t;
}

class Device {
    int id;

public:
    Device() : Device{currentId()} {}
    explicit Device(int id) noexcept : id{id} {}
    static int currentId() { return createFirstArg(cudaGetDevice); }
    static int count() { return createFirstArg(cudaGetDeviceCount); }
    cudaDeviceProp props() const { return createFirstArg(cudaGetDeviceProperties, id); }
    const Device& setCurrent() const {
        throwIfError(cudaSetDevice(id));
        return *this;
    }
    void synchronize() { throwIfError(::cudaDeviceSynchronize()); }
};

constexpr auto memoryAlignment = 256;
constexpr auto defaultResidentGrids = 16;

inline int residentGrids6x(int minor) {
    switch (minor) {
        case 0:
            return 128;
        case 1:
            return 32;
        case 2:
            return 16;
    }
    return defaultResidentGrids;
}

inline int residentGrids7x(int minor) {
    switch (minor) {
        case 0:
            return 128;
        case 2:
            return 16;
        case 5:
            return 128;
    }
    return defaultResidentGrids;
}

inline int residentGrids8x(int minor) { return 128; }

inline int residentGrids(const cudaDeviceProp& p) {
    switch (p.major) {
        case 6:
            return residentGrids6x(p.minor);
        case 7:
            return residentGrids7x(p.minor);
        case 8:
            return residentGrids8x(p.minor);
    }
    return defaultResidentGrids;
}

inline int maxConcurrentStreams(CUDA::Device d) {
    auto p = d.props();
    int r = p.asyncEngineCount;
    if (!p.concurrentKernels) return r + 1;
    return r + residentGrids(p);
}

inline bool isHalfSupported(CUDA::Device d) {
    const auto computeCompatabilityVersion = std::to_string(d.props().major) + "." + std::to_string(d.props().minor);
    return fp16SupportedArchitecture.count(computeCompatabilityVersion) > 0;
}

inline bool isInt8Supported(CUDA::Device d) {
    const auto computeCompatabilityVersion = std::to_string(d.props().major) + "." + std::to_string(d.props().minor);
    return int8SupportedArchitecture.count(computeCompatabilityVersion) > 0;
}

template <typename T>
class Handle {
public:
    using Native = T;
    using Shared = std::shared_ptr<Native>;

    template <typename R, typename... Args>
    using Construct = R (*)(T*, Args... args);

    template <typename R>
    using Destruct = R (*)(Native);

    virtual ~Handle() = 0;

    explicit operator bool() const { return native_.operator bool(); }

    const Native& get() const noexcept { return *native_; }
    const Shared& get_shared() const noexcept { return native_; }

protected:
    template <typename R, typename... Args>
    Handle(Construct<R, Args...> constructor, Destruct<R> destructor, Args... args) {
        auto native = Native{createFirstArg(constructor, args...)};
        try {
            native_ =
                std::shared_ptr<Native>(std::make_unique<Native>(native).release(), [destructor](const Native* native) {
                    if (destructor) {
                        logIfError(destructor(*native));
                    }
                    delete native;
                });
        } catch (...) {
            if (destructor) {
                logIfError(destructor(native));
            }
            throw;
        }
    }

    template <typename R, typename... Args>
    Handle(Construct<R, Args...> constructor, std::nullptr_t, Args... args) {
        auto native = Native{createFirstArg(constructor, args...)};
        native_ = std::shared_ptr<Native>(std::make_unique<Native>(native).release());
    }

private:
    std::shared_ptr<Native> native_;
};

template <typename T>
inline Handle<T>::~Handle() {}

class DefaultAllocation {
    struct Deleter {
        void operator()(void* p) const noexcept { logIfError(cudaFree(p)); }
    };
    std::shared_ptr<void> p;

public:
    explicit DefaultAllocation(void* p) noexcept : p{p, Deleter{}} {}
    void* get() const noexcept { return p.get(); }
    template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
    operator DevicePointer<T*>() const noexcept {
        return DevicePointer<T*>{get()};
    }
};

class Allocation {
    class Deleter {
        Handle<cudaStream_t>::Shared stream;

        auto freeImpl(void* p) const noexcept {
#if CUDART_VERSION >= 11020
            return cudaFreeAsync(p, *stream);
#else
            return cudaFree(p);
#endif
        }

    public:
        Deleter(const Handle<cudaStream_t>& stream) noexcept : stream{stream.get_shared()} {}
        void operator()(void* p) const noexcept { logIfError(freeImpl(p)); }
    };

    std::shared_ptr<void> p;

public:
    Allocation(void* p, const Handle<cudaStream_t>& stream) noexcept : p{p, Deleter{stream}} {}
    void* get() const noexcept { return p.get(); }
    template <typename T, typename std::enable_if<std::is_void<T>::value>::type* = nullptr>
    operator DevicePointer<T*>() const noexcept {
        return DevicePointer<T*>{get()};
    }
};

class Stream : public Handle<cudaStream_t> {
public:
    Stream() : Handle((cudaStreamCreate), cudaStreamDestroy) {}

    Allocation malloc(std::size_t size) const { return {mallocImpl(size), *this}; }
    void upload(CUDA::DevicePointer<void*> dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void transfer(CUDA::DevicePointer<void*> dst, CUDA::DevicePointer<const void*> src, std::size_t count) const {
        throwIfError(cudaMemcpyAsync(dst.get(), src.get(), count, cudaMemcpyDeviceToDevice, get()));
    }
    void upload(const Allocation& dst, const void* src, std::size_t count) const { uploadImpl(dst.get(), src, count); }
    void download(void* dst, const Allocation& src, std::size_t count) const { downloadImpl(dst, src.get(), count); }
    void download(void* dst, CUDA::DevicePointer<const void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void download(void* dst, CUDA::DevicePointer<void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void memset(const Allocation& dst, int value, std::size_t count) const { memsetImpl(dst.get(), value, count); }
    void memset(CUDA::DevicePointer<void*> dst, int value, std::size_t count) const {
        memsetImpl(dst.get(), value, count);
    }
    void synchronize() const { throwIfError(cudaStreamSynchronize(get())); }
#ifdef __CUDACC__
    template <typename... Args>
    void run(dim3 gridDim, dim3 blockDim, void (*kernel)(Args...), Args... args) const {
        kernel
#ifndef __CDT_PARSER__
            <<<gridDim, blockDim, 0, get()>>>
#endif
            (args...);
    }
#endif

private:
    void uploadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, get()));
    }
    void downloadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, get()));
    }
    void* mallocImpl(std::size_t size) const {
        return createFirstArg<void*, cudaError_t>(
#if CUDART_VERSION >= 11020
            cudaMallocAsync, size, get()
#else
            cudaMalloc, size
#endif
        );
    }
    void memsetImpl(void* dst, int value, size_t count) const {
        throwIfError(cudaMemsetAsync(dst, value, count, get()));
    }
};

class DefaultStream {
    void uploadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    }
    void downloadImpl(void* dst, const void* src, std::size_t count) const {
        throwIfError(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    }
    void memsetImpl(void* dst, int value, std::size_t count) const { throwIfError(cudaMemset(dst, value, count)); }
    DefaultStream() = default;

public:
    static DefaultStream& stream() {
        static DefaultStream stream{};
        return stream;
    }

    auto malloc(std::size_t size) const {
        return DefaultAllocation{createFirstArg<void*, cudaError_t>(cudaMalloc, size)};
    }
    void upload(DevicePointer<void*> dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void upload(const DefaultAllocation& dst, const void* src, std::size_t count) const {
        uploadImpl(dst.get(), src, count);
    }
    void download(void* dst, const DefaultAllocation& src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void download(void* dst, DevicePointer<const void*> src, std::size_t count) const {
        downloadImpl(dst, src.get(), count);
    }
    void memset(const DefaultAllocation& dst, int value, std::size_t count) const { memsetImpl(dst.get(), value, count); }
    void memset(CUDA::DevicePointer<void*> dst, int value, std::size_t count) const {
        memsetImpl(dst.get(), value, count);
    }
};

}  // namespace CUDA
