// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view

sealed interface AppUiEffect {
    data class Message(val text: String) : AppUiEffect
    data object OpenAccountSettings : AppUiEffect
}
