// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <details/ie_exception.hpp>

#include <ngraph/function.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include "opset/opset.hpp"


namespace ngraph {
namespace element {
    template <>
    Type from<half_float::half>();
}  //  namespace element
}  //  namespace ngraph
namespace ArmPlugin {
struct NCHW {enum {N, C, H, W, DIMS};};
struct WEIGHTS {enum {C_OUT, C_IN, K_H, K_W};};
struct D2 {enum D2_e{H, W};};

arm_compute::TensorShape ShapeCast(const ngraph::Shape& shape);
arm_compute::DataType DataTypeCast(const ngraph::element::Type type);
std::size_t AxisCast(const std::size_t axis, const std::size_t shapeSize);

template<typename Arg>
struct Argument {
    operator Arg() {
        return _arg;
    }
    template<typename T, typename = std::enable_if_t<std::is_constructible<T, Arg>::value>>
    operator T() {
        return T{_arg};
    }
    Arg _arg;
 };

template<>
struct Argument<arm_compute::ITensor*> {
    enum class Type : bool {Input, Output};
    Argument(arm_compute::ITensor* tensor, Type type) :
        _type{type},
        _tensor{tensor} {
        if (_tensor->info()->has_padding()) {
            _notPaddedTensor.allocator()->init({_tensor->info()->tensor_shape(), 1, _tensor->info()->data_type()});
        }
    }
    template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_same<half_float::half, T>::value>>
    operator T*() {
        if (_tensor->info()->has_padding()) {
            return static_cast<T*>(static_cast<void*>(_notPaddedTensor.buffer()));
        } else {
            return static_cast<T*>(static_cast<void*>(_tensor->buffer()));
        }
    }

    template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_same<half_float::half, T>::value>>
    operator const T*() const {
        return const_cast<Argument<arm_compute::ITensor*>*>(this)->operator T*();
    }
    Type                    _type;
    arm_compute::ITensor*   _tensor;
    arm_compute::Tensor     _notPaddedTensor;
};


struct Converter;
template<typename Arg>
struct ConversionArg {
    operator Arg() {
        return _arg;
    }
    template<typename T, typename = std::enable_if_t<std::is_constructible<T, Arg>::value>>
    operator T() {
        return T{_arg};
    }
    Converter& _converter;
    Arg _arg;
};

template<std::size_t, typename> struct FunctionArgument;

template <std::size_t Index, typename ClassType, typename ReturnType, typename... Arguments>
struct FunctionArgument<Index, ReturnType(ClassType::*)(Arguments...) const> {
    using type = std::tuple_element_t<Index, std::tuple<Arguments...>>;
};

template <std::size_t Index, typename ReturnType, typename... Arguments>
struct FunctionArgument<Index, ReturnType(*)(Arguments...)> {
    using type = std::tuple_element_t<Index, std::tuple<Arguments...>>;
};

template <std::size_t Index, typename ReturnType, typename... Arguments>
struct FunctionArgument<Index, ReturnType(*&)(Arguments...)> {
    using type = std::tuple_element_t<Index, std::tuple<Arguments...>>;
};

struct Layer {
    using Map = std::unordered_map<std::string, Layer>;
    std::unique_ptr<arm_compute::IFunction>             _function;
    std::vector<arm_compute::Tensor*>                   _inputs;
    std::vector<std::unique_ptr<arm_compute::Tensor>>   _outputs;
};

static const std::string& GetNodeName(const ngraph::Input<const ngraph::Node>& input) {
    return input.get_node()->get_friendly_name();
}
static const std::string& GetNodeName(const std::vector<ngraph::Input<const ngraph::Node>>& inputs) {
    return inputs.front().get_node()->get_friendly_name();
}
static const std::string& GetNodeName(const ngraph::Output<const ngraph::Node>& output) {
    return output.get_node()->get_friendly_name();
}
static const std::string& GetNodeName(const std::vector<ngraph::Output<const ngraph::Node>>& outputs) {
    return outputs.front().get_node()->get_friendly_name();
}

template<typename ACFunction, bool Flag>
struct MakeFunction;

template<typename ACFunction>
struct MakeFunction<ACFunction, true> {
    static auto Make(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager) {
        return std::make_unique<ACFunction>(memoryManager);
    }
};

template<typename ACFunction>
struct MakeFunction<ACFunction, false> {
    static auto Make(const std::shared_ptr<arm_compute::IMemoryManager>&) {
        return std::make_unique<ACFunction>();
    }
};

struct Converter {
    struct Conversion {
        using Ptr = std::unique_ptr<Conversion>;
        virtual ~Conversion() = default;
        virtual arm_compute::Status Validate() = 0;
        virtual void Configure(const std::shared_ptr<arm_compute::IMemoryManager>&) = 0;
    };
    template<typename ACFunction, typename... Args>
    struct ConversionImpl final : public Conversion {
        ConversionImpl(Converter& converter, Args&& ... args) :
            _converter{converter},
            _args{std::forward<Args>(args)...} {
        }
        template<typename Arg>
        auto MakeConversionArg(Arg&& arg) {
            return ConversionArg<Arg>{_converter, std::forward<Arg>(arg)};
        }
        struct HasValidate {
            template <typename C> static char test(decltype(&C::validate));
            template <typename C> static long test(...);
            constexpr static const bool value = sizeof(test<ACFunction>(nullptr)) == sizeof(char);
        };
        template<std::size_t... I>
        arm_compute::Status ValidateImpl(std::true_type, std::index_sequence<I...>) {
            return ACFunction::validate(MakeConversionArg(std::get<I>(_args))...);
        }
        template<std::size_t... I>
        arm_compute::Status ValidateImpl(std::false_type, std::index_sequence<I...>) {
            return {};
        }
        arm_compute::Status Validate() override {
            return ValidateImpl(std::integral_constant<bool, HasValidate::value>{},
                                std::make_index_sequence<sizeof...(Args)>{});
        }

        template<std::size_t... I>
        void ConfigureImpl(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager, std::index_sequence<I...>) {
            auto function = MakeFunction<ACFunction,
                std::is_constructible<ACFunction, std::shared_ptr<arm_compute::IMemoryManager>>::value>::Make(memoryManager);
            function->configure(MakeConversionArg(std::get<I>(_args))...);
            _converter._layers.at(GetNodeName(std::get<0>(_args)))._function = std::move(function);
        }
        void Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager) override {
            ConfigureImpl(memoryManager, std::make_index_sequence<sizeof...(Args)>{});
        }
        Converter&                          _converter;
        std::tuple<std::decay_t<Args>...>   _args;
    };

    template<typename ACFunction, typename ...Args>
    Conversion::Ptr MakeConversion(Args&& ... args) {
        return std::make_unique<ConversionImpl<ACFunction, Args...>>(*this, std::forward<Args>(args)...);
    }

    template<typename Callable, typename... Args>
    struct ConversionCallableImpl final : public Conversion {
        ConversionCallableImpl(Converter& converter, Callable&& callable, Args&& ... args) :
            _converter{converter},
            _callable{std::forward<Callable>(callable)},
            _args{std::forward<Args>(args)...} {
        }

        template<typename ... RunArgs>
        struct CallableFunction final : public arm_compute::IFunction {
            template<typename T>
            void StartArgumentLifeTime(T&) {}

            void StartArgumentLifeTime(Argument<arm_compute::ITensor*>& tensorArgument) {
                if (tensorArgument._tensor->info()->has_padding()) {
                    _memoryGroup.manage(&(tensorArgument._notPaddedTensor));
                }
            }

            void StartArgumentLifeTimeImpl() {}

            template<typename H, typename ... T>
            void StartArgumentLifeTimeImpl(H&& h, T&& ... t) {
                StartArgumentLifeTime(std::forward<H>(h));
                StartArgumentLifeTimeImpl(std::forward<T>(t)...);
            }

            template<std::size_t ... I>
            void StartLifeTime(std::index_sequence<I...>) {
                StartArgumentLifeTimeImpl(std::get<I>(_args)...);
            }

            template<typename T>
            void EndArgumentLifeTime(T&&) {}

            void EndArgumentLifeTime(Argument<arm_compute::ITensor*>& tensorArgument) {
                if (tensorArgument._tensor->info()->has_padding()) {
                    tensorArgument._notPaddedTensor.allocator()->allocate();
                }
            }

            template<typename H>
            void EndArgumentLifeTimeImpl(H&& h) {
                EndArgumentLifeTime(std::forward<H>(h));
            }

            template<typename H, typename ... T>
            void EndArgumentLifeTimeImpl(H&& h, T&& ... t) {
                EndArgumentLifeTime(std::forward<H>(h));
                EndArgumentLifeTimeImpl(std::forward<T>(t)...);
            }

            template<std::size_t ... I>
            void EndLifeTime(std::index_sequence<I...>) {
                EndArgumentLifeTimeImpl(std::get<I>(_args)...);
            }

            CallableFunction(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager,
                             Callable&& callable,
                             RunArgs&& ... args) :
                _memoryGroup{memoryManager},
                _callable{std::forward<Callable>(callable)},
                _args{std::forward<RunArgs>(args)...} {
                StartLifeTime(std::make_index_sequence<sizeof...(RunArgs)>{});
                EndLifeTime(std::make_index_sequence<sizeof...(RunArgs)>{});
            }

            template<typename T>
            void CopyArgument(Argument<arm_compute::ITensor*>::Type, T&&) {}

            void CopyArgument(Argument<arm_compute::ITensor*>::Type type, Argument<arm_compute::ITensor*>& tensorArgument) {
                if (tensorArgument._tensor->info()->has_padding()) {
                    if (tensorArgument._type == type) {
                        switch (type) {
                        case Argument<arm_compute::ITensor*>::Type::Input  : tensorArgument._notPaddedTensor.copy_from(*(tensorArgument._tensor)); break;
                        case Argument<arm_compute::ITensor*>::Type::Output : tensorArgument._tensor->copy_from(tensorArgument._notPaddedTensor); break;
                        }
                    }
                }
            }

            void CopyArguments(Argument<arm_compute::ITensor*>::Type type) {}

            template<typename H, typename ... T>
            void CopyArguments(Argument<arm_compute::ITensor*>::Type type, H&& h, T&& ... t) {
                CopyArgument(type, std::forward<H>(h));
                CopyArguments(type, std::forward<T>(t)...);
            }

            template<std::size_t ... I>
            void RunImpl(std::index_sequence<I...>) {
                CopyArguments(Argument<arm_compute::ITensor*>::Type::Input, std::get<I>(_args)...);
                _callable(std::get<I>(_args)...);
                CopyArguments(Argument<arm_compute::ITensor*>::Type::Output, std::get<I>(_args)...);
            }
            void run() override {
                RunImpl(std::make_index_sequence<sizeof...(RunArgs)>{});
            }

            arm_compute::MemoryGroup                        _memoryGroup;
            std::decay_t<Callable>                          _callable;
            std::tuple<std::decay_t<RunArgs>...>            _args;
        };

        template<std::size_t I, typename T>
        decltype(auto) MakeArgument(T&& x) {
            static_assert(std::is_same<std::decay_t<T>, std::decay_t<typename FunctionArgument<I, Callable>::type>>::value,
                "Arguments type should be the same");
            return x;
        }

        template<std::size_t I>
        std::nullptr_t MakeArgument(std::nullptr_t x) {
            return nullptr;
        }

        template<std::size_t I>
        Argument<arm_compute::ITensor*> MakeArgument(ngraph::Input<const ngraph::Node>& input) {
            auto type = ngraph::element::from<std::remove_const_t<std::remove_pointer_t<std::decay_t<typename FunctionArgument<I, Callable>::type>>>>();
            if (input.get_element_type() != type) {
                THROW_IE_EXCEPTION << "Argument types should be the same "
                    << input.get_element_type() << " "
                    << type;
            }
            return {_converter._layers.at(input.get_node()->get_friendly_name())._inputs.at(input.get_index()),
                    Argument<arm_compute::ITensor*>::Type::Input};
        }

        template<std::size_t I>
        Argument<arm_compute::ITensor*> MakeArgument(ngraph::Output<const ngraph::Node>& output) {
            auto type = ngraph::element::from<std::remove_const_t<std::remove_pointer_t<std::decay_t<typename FunctionArgument<I, Callable>::type>>>>();
            if (output.get_element_type() != type) {
                THROW_IE_EXCEPTION << "Argument types should be the same "
                    << output.get_element_type() << " "
                    << type;
            }
            return {_converter._layers.at(output.get_node()->get_friendly_name())._outputs.at(output.get_index()).get(),
                    Argument<arm_compute::ITensor*>::Type::Output};
        }

        arm_compute::Status Validate() override {
            return {};
        }

        template<typename ... RunArgs>
        auto makeCallableFunction(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager, Callable&& callable, RunArgs&& ... args) {
            return std::make_unique<CallableFunction<RunArgs...>>(
                memoryManager, std::forward<Callable>(callable), std::forward<RunArgs>(args)...);
        }

        template<std::size_t... I>
        void ConfigureImpl(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager, std::index_sequence<I...>) {
            auto function = makeCallableFunction(memoryManager, _callable, MakeArgument<I>(std::get<I>(_args))...);
            _converter._layers.at(GetNodeName(std::get<0>(_args)))._function = std::move(function);
        }
        void Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager) override {
            ConfigureImpl(memoryManager, std::make_index_sequence<sizeof...(Args)>{});
        }

        Converter&                          _converter;
        std::decay_t<Callable>              _callable;
        std::tuple<std::decay_t<Args>...>   _args;
    };

    template<typename Callable, typename... Args>
    Conversion::Ptr MakeConversion(Callable&& callable, Args&& ... args) {
        return std::make_unique<ConversionCallableImpl<Callable, Args...>>(*this, std::forward<Callable>(callable), std::forward<Args>(args)...);
    }

    Converter(const std::shared_ptr<const ngraph::Function> function, bool ref = true);

    Layer::Map Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager,
                         arm_compute::MemoryGroup& memoryGroup);

    template<typename NodeType>
    Conversion::Ptr Convert(const NodeType& node);

    using ConvertFn = std::function<Conversion::Ptr(const ngraph::Node&)>;
    template<typename NodeType>
    void Register() {
        _conversions.emplace(NodeType::type_info,
            [this] (const ngraph::Node& node) {return Convert(dynamic_cast<const NodeType&>(node));});
    }

    std::map<ngraph::Node::type_info_t, ConvertFn>  _conversions;
    std::shared_ptr<const ngraph::Function>         _function;
    Layer::Map                                      _layers;
};

template<>
struct ConversionArg<ngraph::Input<const ngraph::Node>&> {
    operator arm_compute::ITensorInfo*() {
        return _converter._layers.at(_input.get_node()->get_friendly_name())._inputs.at(_input.get_index())->info();
    }
    operator arm_compute::ITensor*() {
        return _converter._layers.at(_input.get_node()->get_friendly_name())._inputs.at(_input.get_index());
    }
    Converter&                    _converter;
    ngraph::Input<const ngraph::Node>&  _input;
};

template<>
struct ConversionArg<ngraph::Output<const ngraph::Node>&> {
    operator arm_compute::ITensorInfo*() {
        return _converter._layers.at(_output.get_node()->get_friendly_name())._outputs.at(_output.get_index())->info();
    }
    operator arm_compute::ITensor*() {
        return _converter._layers.at(_output.get_node()->get_friendly_name())._outputs.at(_output.get_index()).get();
    }
    Converter&                    _converter;
    ngraph::Output<const ngraph::Node>&  _output;
};

template<>
struct ConversionArg<std::vector<ngraph::Input<const ngraph::Node>>&> {
    operator std::vector<const arm_compute::ITensorInfo*>() const {
        std::vector<const arm_compute::ITensorInfo*> infos;
        for (auto&& input : _inputs) {
            infos.emplace_back(_converter._layers.at(input.get_node()->get_friendly_name())._inputs.at(input.get_index())->info());
        }
        return infos;
    }
    operator std::vector<const arm_compute::ITensor*>() const {
        std::vector<const arm_compute::ITensor*> tensors;
        for (auto&& input : _inputs) {
            tensors.emplace_back(_converter._layers.at(input.get_node()->get_friendly_name())._inputs.at(input.get_index()));
        }
        return tensors;
    }
    Converter&                                      _converter;
    std::vector<ngraph::Input<const ngraph::Node>>& _inputs;
};
template<>
struct ConversionArg<std::vector<ngraph::Output<const ngraph::Node>>&> {
    operator std::vector<arm_compute::ITensorInfo*>() const {
        std::vector<arm_compute::ITensorInfo*> infos;
        for (auto&& output : _outputs) {
            infos.emplace_back(_converter._layers.at(output.get_node()->get_friendly_name())._outputs.at(output.get_index())->info());
        }
        return infos;
    }
    operator std::vector<arm_compute::ITensor*>() const {
        std::vector<arm_compute::ITensor*> tensors;
        for (auto&& output : _outputs) {
            tensors.emplace_back(_converter._layers.at(output.get_node()->get_friendly_name())._outputs.at(output.get_index()).get());
        }
        return tensors;
    }
    Converter&                                          _converter;
    std::vector<ngraph::Output<const ngraph::Node>>&    _outputs;
};

template<>
struct ConversionArg<std::pair<ngraph::Input<const ngraph::Node>, arm_compute::TensorInfo>&> {
    enum {NodeInput, TensorInfo};
    operator arm_compute::ITensorInfo*() const {
        return &(std::get<TensorInfo>(_input));
    }
    operator arm_compute::ITensor*() const {
        auto& input = std::get<NodeInput>(_input);
        auto sourceOutput = input.get_source_output();
        auto& tensor = _converter._layers.at(sourceOutput.get_node()->get_friendly_name())._outputs.at(sourceOutput.get_index());
        static_cast<arm_compute::Tensor*>(tensor.get())->allocator()->init(std::get<TensorInfo>(_input));
        return tensor.get();
    }
    Converter&                                                              _converter;
    std::pair<ngraph::Input<const ngraph::Node>, arm_compute::TensorInfo>&  _input;
};

}  //  namespace ArmPlugin
