bool try_build_eltwise(const std::shared_ptr<const ov::Model>& model,
                           const ModelAnalysis& analysis,
                           MetalKernelCompiler& compiler,
                           std::string& log);
    bool try_build_softmax(const std::shared_ptr<const ov::Model>& model,
                           const ModelAnalysis& analysis,
                           MetalKernelCompiler& compiler,
                           std::string& log);
    bool try_build_matmul(const std::shared_ptr<const ov::Model>& model,
                          const ModelAnalysis& analysis,
                          MetalKernelCompiler& compiler,
                          std::string& log);
    bool try_build_conv(const std::shared_ptr<const ov::Model>& model,
                        const ModelAnalysis& analysis,
                        MetalKernelCompiler& compiler,
                        std::string& log);
    bool try_build_pool(const std::shared_ptr<const ov::Model>& model,
                        const ModelAnalysis& analysis,
                        MetalKernelCompiler& compiler,
                        std::string& log);
    bool try_build_unary(const std::shared_ptr<const ov::Model>& model,
                         const ModelAnalysis& analysis,
                         MetalKernelCompiler& compiler,
                         std::string& log);
    bool try_build_batchnorm(const std::shared_ptr<const ov::Model>& model,
                             const ModelAnalysis& analysis,
                             MetalKernelCompiler& compiler,
                             std::string& log);
    bool try_run_split_segment(const Segment& seg,
                               const std::vector<KernelOp>& ops,
                               const std::vector<ov::Tensor>& inputs,
                               std::vector<ov::Tensor>& outputs,
                               MetalBufferManager& mgr,
                               const std::function<void(const std::string&)>& cpu_fallback_error);
