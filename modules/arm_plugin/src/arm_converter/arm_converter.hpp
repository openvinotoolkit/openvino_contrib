// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/util/pp.hpp>
#include <ie_common.h>
#include <ie_algorithm.hpp>
#include <ngraph/function.hpp>

#include <arm_compute/runtime/IFunction.h>
#include <arm_compute/runtime/Tensor.h>

#include "arm_config.hpp"
#include "opset/opset.hpp"


namespace ArmPlugin {
struct NCHW {enum {N, C, H, W, DIMS};};
struct WEIGHTS {enum {C_OUT, C_IN, K_H, K_W};};
struct D2 {enum D2_e{H, W};};

arm_compute::TensorShape ShapeCast(const ngraph::Shape& shape);
arm_compute::DataType DataTypeCast(const ngraph::element::Type type);
std::size_t AxisCast(const std::size_t axis, const std::size_t shapeSize);

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

template<std::size_t Index, typename T> struct FunctionArgument : public FunctionArgument<Index, decltype(&T::operator())> {};

template <std::size_t Index, typename ClassType, typename ReturnType, typename... Arguments>
struct FunctionArgument<Index, ReturnType(ClassType::*)(Arguments...)> {
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

struct Input : ngraph::Input<ngraph::Node> {
    Input(const ngraph::Input<ngraph::Node>& input) : ngraph::Input<ngraph::Node>{input} {}
    Input(const ngraph::Input<const ngraph::Node>& input) :
    Input{reinterpret_cast<const ngraph::Input<ngraph::Node>&>(input)} {}
};

struct Output : ngraph::Output<ngraph::Node> {
    Output(const ngraph::Output<ngraph::Node>& output) : ngraph::Output<ngraph::Node>{output} {}
    Output(const ngraph::Output<const ngraph::Node>& output) :
    Output{reinterpret_cast<const ngraph::Output<ngraph::Node>&>(output)} {}
};

struct Tensor {
    std::unique_ptr<arm_compute::Tensor>    _tensor;
    std::unique_ptr<arm_compute::Tensor>    _notPaddedTensor;
};

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

enum class ArgumentType : bool {Input, Output};
template<>
struct Argument<Tensor*> {
    Argument(Tensor* tensor, ArgumentType type) :
        _type{type},
        _tensor{tensor} {
        _tensor->_notPaddedTensor = std::make_unique<arm_compute::Tensor>();
    }
    template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_same<ngraph::float16, T>::value>>
    operator T*() {
        if (_tensor->_tensor->info()->has_padding()) {
            return static_cast<T*>(static_cast<void*>(_tensor->_notPaddedTensor->buffer()));
        } else {
            return static_cast<T*>(static_cast<void*>(_tensor->_tensor->buffer()));
        }
    }

    template<typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_same<ngraph::float16, T>::value>>
    operator const T*() const {
        return const_cast<Argument<Tensor*>*>(this)->operator T*();
    }
    ArgumentType            _type;
    Tensor*                 _tensor;
};

struct HostTensors {
    ngraph::HostTensorVector _hosts;
    const ngraph::Node*      _node = nullptr;
};

template<>
struct Argument<HostTensors> {
    Argument(const HostTensors& hosts, const std::vector<Tensor*>& tensors, ArgumentType type) :
        _type{type},
        _tensors(tensors),
        _hosts{hosts._hosts} {
        for (auto tensor : _tensors) {
            tensor->_notPaddedTensor = std::make_unique<arm_compute::Tensor>();
        }
    }
    operator ngraph::HostTensorVector() {
        return _hosts;
    }

    ArgumentType                _type;
    std::vector<Tensor*>        _tensors;
    ngraph::HostTensorVector    _hosts;
};

struct Layer {
    using Map = std::unordered_map<std::size_t, Layer>;
    std::unique_ptr<arm_compute::IFunction>     _function;
    std::map<Input, Tensor*>                    _inputs;
    std::map<Output, Tensor>                    _outputs;
    std::string                                 _execType;
};

static std::size_t GetNodeId(const ngraph::Input<const ngraph::Node>& input) {
    return input.get_node()->get_instance_id();
}
static std::size_t GetNodeId(const std::vector<ngraph::Input<const ngraph::Node>>& inputs) {
    return inputs.front().get_node()->get_instance_id();
}
static std::size_t GetNodeId(const ngraph::Output<const ngraph::Node>& output) {
    return output.get_node()->get_instance_id();
}
static std::size_t GetNodeId(const std::vector<ngraph::Output<const ngraph::Node>>& outputs) {
    return outputs.front().get_node()->get_instance_id();
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
        virtual std::string ExecType() const = 0;
    };
    template<typename ACFunction, typename... Args>
    struct ConversionImpl final : public Conversion {
        ConversionImpl(Converter& converter, Args&& ... args) :
            _converter{converter},
            _args{std::forward<Args>(args)...} {
        }
        std::string ExecType() const override {
            return "Arm Compute";
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
            _converter._layers.at(GetNodeId(std::get<0>(_args)))._function = std::move(function);
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

        std::string ExecType() const override {
            return "ngraph Reference";
        }

        template<typename ... RunArgs>
        struct CallableFunction final : public arm_compute::IFunction {
            CallableFunction(std::decay_t<Callable>& callable,
                             RunArgs&& ... args) :
                _callable{callable},
                _args{std::forward<RunArgs>(args)...} {
            }

            template<typename T>
            void CopyArgument(ArgumentType, T&&) {}

            void CopyArgument(ArgumentType type, Argument<Tensor*>& tensorArgument) {
                if (tensorArgument._tensor->_tensor->info()->has_padding()) {
                    if (tensorArgument._type == type) {
                        switch (type) {
                        case ArgumentType::Input  : tensorArgument._tensor->_notPaddedTensor->copy_from(*(tensorArgument._tensor->_tensor)); break;
                        case ArgumentType::Output : tensorArgument._tensor->_tensor->copy_from(*(tensorArgument._tensor->_notPaddedTensor)); break;
                        }
                    }
                }
            }

            void CopyArgument(ArgumentType type, Argument<HostTensors>& hostsArgument) {
                for (std::size_t i = 0; i < hostsArgument._hosts.size(); i++) {
                    void* host_ptr = static_cast<void*>(hostsArgument._hosts[i]->get_data_ptr());
                    void* tensor_ptr = static_cast<void*>(hostsArgument._tensors[i]->_tensor->buffer());
                    if (host_ptr != tensor_ptr) {
                        hostsArgument._hosts[i] = std::make_shared<ngraph::HostTensor>(
                                                    hostsArgument._hosts[i]->get_element_type(),
                                                    hostsArgument._hosts[i]->get_shape(),
                                                    tensor_ptr);
                    }
                    if (hostsArgument._tensors[i]->_tensor->info()->has_padding()) {
                        if (hostsArgument._type == type) {
                            switch (type) {
                            case ArgumentType::Input  :
                                hostsArgument._tensors[i]->_notPaddedTensor->copy_from(*(hostsArgument._tensors[i]->_tensor));
                                break;
                            case ArgumentType::Output :
                                hostsArgument._tensors[i]->_tensor->copy_from(*(hostsArgument._tensors[i]->_notPaddedTensor));
                                break;
                            }
                        }
                    }
                }
            }

            void CopyArguments(ArgumentType type) {}

            template<typename H, typename ... T>
            void CopyArguments(ArgumentType type, H&& h, T&& ... t) {
                CopyArgument(type, std::forward<H>(h));
                CopyArguments(type, std::forward<T>(t)...);
            }

            template<std::size_t ... I>
            void RunImpl(std::index_sequence<I...>) {
                CopyArguments(ArgumentType::Input, std::get<I>(_args)...);
                _callable(std::get<I>(_args)...);
                CopyArguments(ArgumentType::Output, std::get<I>(_args)...);
            }
            void run() override {
                RunImpl(std::make_index_sequence<sizeof...(RunArgs)>{});
            }

            std::decay_t<Callable>                          _callable;
            std::tuple<std::decay_t<RunArgs>...>            _args;
        };

        template<std::size_t I, typename T>
        decltype(auto) MakeArgument(T&& x) {
            static_assert(std::is_same<std::decay_t<T>, std::decay_t<typename FunctionArgument<I, std::decay_t<Callable>>::type>>::value,
                "Arguments type should be the same");
            return x;
        }

        template<std::size_t I>
        std::nullptr_t MakeArgument(std::nullptr_t x) {
            return nullptr;
        }

        template<std::size_t I>
        Argument<Tensor*> MakeArgument(ngraph::Input<const ngraph::Node>& input) {
            auto type = ngraph::element::from<
                std::remove_const_t<std::remove_pointer_t<std::decay_t<typename FunctionArgument<I, std::decay_t<Callable>>::type>>>>();
            if (input.get_element_type() != type) {
                IE_THROW() << "Argument types should be the same " << input << " " << type;
            }
            return {_converter._layers.at(input.get_node()->get_instance_id())._inputs.at(input), ArgumentType::Input};
        }

        template<std::size_t I>
        Argument<Tensor*> MakeArgument(ngraph::Output<const ngraph::Node>& output) {
            auto type = ngraph::element::from<
                std::remove_const_t<std::remove_pointer_t<std::decay_t<typename FunctionArgument<I, std::decay_t<Callable>>::type>>>>();
            if (output.get_element_type() != type) {
                IE_THROW() << "Argument types should be the same " << output << " " << type;
            }
            return {&(_converter._layers.at(output.get_node()->get_instance_id())._outputs.at(output)), ArgumentType::Output};
        }

        template<std::size_t I>
        Argument<HostTensors> MakeArgument(HostTensors& hosts) {
            std::vector<Tensor*> tensors;
            IE_ASSERT(hosts._node != nullptr);
            for (auto&& output : hosts._node->outputs()) {
                tensors.push_back(&(_converter._layers.at(hosts._node->get_instance_id())._outputs.at(output)));
            }
            IE_ASSERT(tensors.size() == hosts._hosts.size());
            return {hosts, tensors, ArgumentType::Output};
        }

        template<std::size_t I>
        std::vector<Argument<Tensor*>> MakeArgument(std::vector<ngraph::Input<const ngraph::Node>>& inputs) {
            std::vector<Argument<Tensor*>> tensors;
            for (const auto& input : inputs) {
                tensors.emplace_back(_converter._layers.at(input.get_node()->get_instance_id())._inputs.at(input), ArgumentType::Input);
            }
            return tensors;
        }

        template<std::size_t I>
        std::vector<Argument<Tensor*>> MakeArgument(std::vector<ngraph::Output<const ngraph::Node>>& outputs) {
            std::vector<Argument<Tensor*>> tensors;
            for (const auto& output : outputs) {
                tensors.emplace_back(&(_converter._layers.at(output.get_node()->get_instance_id())._outputs.at(output)), ArgumentType::Output);
            }
            return tensors;
        }

        arm_compute::Status Validate() override {
            return {};
        }

        template<typename ... RunArgs>
        auto makeCallableFunction(std::decay_t<Callable>& callable, RunArgs&& ... args) {
            return std::make_unique<CallableFunction<RunArgs...>>(callable, std::forward<RunArgs>(args)...);
        }

        template<std::size_t... I>
        void ConfigureImpl(std::index_sequence<I...>) {
            auto function = makeCallableFunction(_callable, MakeArgument<I>(std::get<I>(_args))...);
            _converter._layers.at(GetNodeId(std::get<0>(_args)))._function = std::move(function);
        }
        void Configure(const std::shared_ptr<arm_compute::IMemoryManager>&) override {
            ConfigureImpl(std::make_index_sequence<sizeof...(Args)>{});
        }

        Converter&                          _converter;
        std::decay_t<Callable>              _callable;
        std::tuple<std::decay_t<Args>...>   _args;
    };

    template<typename Callable, typename... Args>
    Conversion::Ptr MakeConversion(Callable&& callable, Args&& ... args) {
        return std::make_unique<ConversionCallableImpl<Callable, Args...>>(*this, std::forward<Callable>(callable), std::forward<Args>(args)...);
    }

    Converter(const std::shared_ptr<const ov::Model> model, const Configuration& cfg);

    Layer::Map Configure(const std::shared_ptr<arm_compute::IMemoryManager>& memoryManager,
                         arm_compute::MemoryGroup& memoryGroup);

    template<typename NodeType>
    Conversion::Ptr Convert(const NodeType& node);

    using ConvertFn = std::function<Conversion::Ptr(const ngraph::Node&)>;
    template<typename NodeType>
    void Register() {
        _conversions.emplace(NodeType::get_type_info_static(), [this] (const ngraph::Node& node) {
            OPENVINO_ASSERT(ov::is_type<NodeType>(&node),
                " node: ", node,
                " current type_info: ", NodeType::get_type_info_static());
            return Convert(static_cast<const NodeType&>(node));
        });
    }

    const Configuration                             _cfg;
    std::map<ngraph::Node::type_info_t, ConvertFn>  _conversions;
    std::shared_ptr<const ov::Model>                _model;
    Layer::Map                                      _layers;
};

template<>
struct ConversionArg<ngraph::Input<const ngraph::Node>&> {
    operator arm_compute::ITensorInfo*() {
        return _converter._layers.at(_input.get_node()->get_instance_id())._inputs.at(_input)->_tensor->info();
    }
    operator arm_compute::ITensor*() {
        return _converter._layers.at(_input.get_node()->get_instance_id())._inputs.at(_input)->_tensor.get();
    }
    Converter&                    _converter;
    ngraph::Input<const ngraph::Node>&  _input;
};

template<>
struct ConversionArg<ngraph::Output<const ngraph::Node>&> {
    operator arm_compute::ITensorInfo*() {
        return _converter._layers.at(_output.get_node()->get_instance_id())._outputs.at(_output)._tensor->info();
    }
    operator arm_compute::ITensor*() {
        return _converter._layers.at(_output.get_node()->get_instance_id())._outputs.at(_output)._tensor.get();
    }
    Converter&                    _converter;
    ngraph::Output<const ngraph::Node>&  _output;
};

template<>
struct ConversionArg<std::vector<ngraph::Input<const ngraph::Node>>&> {
    operator std::vector<const arm_compute::ITensorInfo*>() const {
        std::vector<const arm_compute::ITensorInfo*> infos;
        for (auto&& input : _inputs) {
            infos.emplace_back(_converter._layers.at(input.get_node()->get_instance_id())._inputs.at(input)->_tensor->info());
        }
        return infos;
    }
    operator std::vector<const arm_compute::ITensor*>() const {
        std::vector<const arm_compute::ITensor*> tensors;
        for (auto&& input : _inputs) {
            tensors.emplace_back(_converter._layers.at(input.get_node()->get_instance_id())._inputs.at(input)->_tensor.get());
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
            infos.emplace_back(_converter._layers.at(output.get_node()->get_instance_id())._outputs.at(output)._tensor->info());
        }
        return infos;
    }
    operator std::vector<arm_compute::ITensor*>() const {
        std::vector<arm_compute::ITensor*> tensors;
        for (auto&& output : _outputs) {
            tensors.emplace_back(_converter._layers.at(output.get_node()->get_instance_id())._outputs.at(output)._tensor.get());
        }
        return tensors;
    }
    Converter&                                          _converter;
    std::vector<ngraph::Output<const ngraph::Node>>&    _outputs;
};

#define AP_WRAP(MAKE, F) [&](auto ... v) {return MAKE(F<decltype(v)...>);}

template<typename IO>
static auto get_element_type(const IO& io) {
    return io.get_element_type();
}

static auto get_element_type(const ngraph::element::Type& type) {
    return type;
}

template<typename, typename ... PT, typename F>
auto CallSwitchPT(std::tuple<int, PT...>, F&& f) {
        return f(PT{} ...);
}

template<typename ReturnT, typename ... PT, typename F, typename IO, typename ... TRest>
[[noreturn]] ReturnT CallSwitchPT(std::tuple<int, PT...>, std::tuple<int>, F&& f, const IO& io, TRest ... args) {
    IE_THROW() << "Unsupported Type: " << io;
}

template<typename ReturnT, typename ... PT, typename F, typename IO, typename ... T, typename ... TRest>
auto CallSwitchPT(std::tuple<int, PT...>, F&&, const IO&, std::tuple<T...>, TRest ... args);

template<typename, typename ... PT, typename H, typename ... T, typename F, typename IO, typename ... TRest>
auto CallSwitchPT(std::tuple<int, PT...>, std::tuple<int, H, T...>, F&& f, const IO& io, TRest ... args) {
    if (ngraph::element::from<H>() == get_element_type(io)) {
        return CallSwitchPT<void>(std::tuple<int, PT ... , H>{}, std::forward<F>(f), args...);
    } else {
        return CallSwitchPT<
            decltype(CallSwitchPT<void>(std::tuple<int, PT ... , H>{}, std::forward<F>(f), args...))
        >(std::tuple<int, PT...>{}, std::tuple<int, T...>{}, std::forward<F>(f), io, args...);
    }
}

template<typename ReturnT, typename ... PT, typename F, typename IO, typename ... T, typename ... TRest>
auto CallSwitchPT(std::tuple<int, PT...>, F&& f, const IO& io, std::tuple<T...>, TRest ... args) {
        return CallSwitchPT<ReturnT>(std::tuple<int, PT...>{}, std::tuple<int, T...>{}, std::forward<F>(f), io, args...);
}

template<typename F, typename IO, typename ... T, typename ... TRest>
auto CallSwitch(F&& f, const IO& io, std::tuple<T...>, TRest ... args) {
        return CallSwitchPT<void>(std::tuple<int>{}, std::forward<F>(f), io, std::tuple<T...>{}, args...);
}

template<typename ...T0, typename ...T1>
constexpr static std::tuple<T0..., T1...> merge(std::tuple<T0...>, std::tuple<T1...>) {return {};}

constexpr static auto boolType = std::tuple<bool>{};
constexpr static auto intTypes = std::tuple<std::int8_t, std::uint8_t, std::int16_t, std::uint16_t, std::int32_t, std::uint32_t, std::int64_t>{};
constexpr static auto indexTypes = std::tuple<std::int32_t, std::int64_t>{};
constexpr static auto floatTypes = std::tuple<ngraph::float16, float>{};
constexpr static auto allTypes = merge(intTypes, floatTypes);
}  //  namespace ArmPlugin
