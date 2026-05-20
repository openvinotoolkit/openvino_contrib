package com.itlab.notes.onboarding

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

@Composable
fun coachMarkOverlay(
    step: OnboardingTourStep,
    stepIndex: Int,
    stepCount: Int,
    targetBounds: Rect?,
    screenMatchesStep: Boolean,
    onSkip: () -> Unit,
    onBack: () -> Unit,
    onNext: () -> Unit,
) {
    val colors = MaterialTheme.colorScheme
    val scrim = Color.Black.copy(alpha = 0.62f)
    val highlightPadding = 10.dp
    val density = LocalDensity.current
    val paddedHole =
        remember(targetBounds, density) {
            targetBounds?.let { bounds ->
                val pad = with(density) { highlightPadding.toPx() }
                Rect(
                    left = bounds.left - pad,
                    top = bounds.top - pad,
                    right = bounds.right + pad,
                    bottom = bounds.bottom + pad,
                )
            }
        }

    Box(modifier = Modifier.fillMaxSize()) {
        if (screenMatchesStep && paddedHole != null) {
            scrimWithHole(
                hole = paddedHole,
                scrimColor = scrim,
            )
        } else {
            Box(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .background(scrim)
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick = {},
                        ),
            )
        }

        Surface(
            modifier =
                Modifier
                    .align(Alignment.BottomCenter)
                    .padding(horizontal = 20.dp, vertical = 28.dp)
                    .fillMaxWidth(),
            shape = RoundedCornerShape(20.dp),
            color = colors.surfaceContainerHigh,
            tonalElevation = 6.dp,
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text(
                    text = "Step ${stepIndex + 1} of $stepCount",
                    style = MaterialTheme.typography.labelMedium,
                    color = colors.onSurfaceVariant,
                )
                Text(
                    text = step.title,
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = colors.onSurface,
                    modifier = Modifier.padding(top = 4.dp),
                )
                Text(
                    text = step.description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = colors.onSurfaceVariant,
                    modifier = Modifier.padding(top = 8.dp),
                )
                if (!screenMatchesStep && step.requiredScreen != null) {
                    Text(
                        text = "Follow the hint on screen to continue.",
                        style = MaterialTheme.typography.bodySmall,
                        color = colors.primary,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
                Row(
                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .padding(top = 16.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextButton(onClick = onSkip) {
                        Text("Skip tour")
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        if (stepIndex > 0) {
                            TextButton(onClick = onBack) {
                                Text("Back")
                            }
                        }
                        Button(onClick = onNext) {
                            Text(if (stepIndex >= stepCount - 1) "Done" else "Next")
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun scrimWithHole(
    hole: Rect,
    scrimColor: Color,
) {
    val interaction = remember { MutableInteractionSource() }
    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
        val widthPx = constraints.maxWidth.toFloat()
        val heightPx = constraints.maxHeight.toFloat()
        val holeLeft = hole.left.coerceIn(0f, widthPx)
        val holeTop = hole.top.coerceIn(0f, heightPx)
        val holeRight = hole.right.coerceIn(holeLeft, widthPx)
        val holeBottom = hole.bottom.coerceIn(holeTop, heightPx)
        val density = LocalDensity.current

        scrimPanel(
            x = 0f,
            y = 0f,
            width = widthPx,
            height = holeTop,
            scrimColor = scrimColor,
            interaction = interaction,
            density = density,
        )
        scrimPanel(
            x = 0f,
            y = holeBottom,
            width = widthPx,
            height = heightPx - holeBottom,
            scrimColor = scrimColor,
            interaction = interaction,
            density = density,
        )
        scrimPanel(
            x = 0f,
            y = holeTop,
            width = holeLeft,
            height = holeBottom - holeTop,
            scrimColor = scrimColor,
            interaction = interaction,
            density = density,
        )
        scrimPanel(
            x = holeRight,
            y = holeTop,
            width = widthPx - holeRight,
            height = holeBottom - holeTop,
            scrimColor = scrimColor,
            interaction = interaction,
            density = density,
        )
    }
}

@Composable
private fun scrimPanel(
    x: Float,
    y: Float,
    width: Float,
    height: Float,
    scrimColor: Color,
    interaction: MutableInteractionSource,
    density: androidx.compose.ui.unit.Density,
) {
    if (width <= 0f || height <= 0f) return
    Box(
        modifier =
            Modifier
                .offset { IntOffset(x.roundToInt(), y.roundToInt()) }
                .width(with(density) { width.toDp() })
                .height(with(density) { height.toDp() })
                .background(scrimColor)
                .clickable(
                    interactionSource = interaction,
                    indication = null,
                    onClick = {},
                ),
    )
}
