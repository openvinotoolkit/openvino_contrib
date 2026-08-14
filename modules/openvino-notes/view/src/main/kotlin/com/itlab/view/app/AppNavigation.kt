// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.view.app

import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable

internal enum class Destination { NOTES, SYNC, SETTINGS, ASSISTANT }

@Composable
internal fun AppNavigation(selected: Destination, onSelected: (Destination) -> Unit) {
    NavigationBar {
        Destination.entries.forEach { item ->
            NavigationBarItem(
                selected = selected == item,
                onClick = { onSelected(item) },
                icon = { Text(item.name.take(1)) },
                label = { Text(item.name.lowercase().replaceFirstChar(Char::uppercase)) },
            )
        }
    }
}
