// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gfx_runtime_model_runner.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace ov::test::gfx {
namespace {

class ForkRuntimeExecutionPolicy final : public RuntimeExecutionPolicy {
public:
    void run(const std::function<void()>& fn, int timeout_seconds) const override {
        const pid_t pid = fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            try {
                fn();
                _exit(0);
            } catch (const std::exception& ex) {
                std::cerr << "child_failure: " << ex.what() << std::endl;
                _exit(1);
            } catch (...) {
                std::cerr << "child_failure: unknown exception" << std::endl;
                _exit(1);
            }
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);
        int status = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            const pid_t waited = waitpid(pid, &status, WNOHANG);
            ASSERT_NE(waited, -1);
            if (waited == pid) {
                if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                    return;
                }
                FAIL() << "subprocess exited with status " << status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        FAIL() << "subprocess timed out after " << timeout_seconds << " seconds";
    }
};

}  // namespace

std::unique_ptr<RuntimeExecutionPolicy> make_runtime_execution_policy() {
    return std::make_unique<ForkRuntimeExecutionPolicy>();
}

}  // namespace ov::test::gfx
