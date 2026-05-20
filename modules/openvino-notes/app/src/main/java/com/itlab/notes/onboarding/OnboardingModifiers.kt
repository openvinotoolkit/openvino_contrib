package com.itlab.notes.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.boundsInRoot
import androidx.compose.ui.layout.onGloballyPositioned

val LocalOnboardingRegistrar =
    staticCompositionLocalOf<((String, Rect?) -> Unit)?> {
        null
    }

@Composable
fun Modifier.onboardingTarget(key: String): Modifier {
    val registrar = LocalOnboardingRegistrar.current ?: return this
    return onGloballyPositioned { coordinates ->
        val bounds = coordinates.boundsInRoot()
        registrar(
            key,
            if (bounds.width > 0f && bounds.height > 0f) bounds else null,
        )
    }
}
