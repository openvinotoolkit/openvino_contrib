package com.itlab.notes.onboarding

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.PagerState
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Cloud
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.Note
import androidx.compose.material.icons.rounded.WavingHand
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.lerp
import kotlinx.coroutines.launch
import kotlin.math.absoluteValue

private data class WelcomeSlide(
    val icon: ImageVector,
    val title: String,
    val body: String,
)

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun welcomeOnboardingScreen(
    onFinished: () -> Unit,
    onSkip: () -> Unit,
) {
    val slides =
        listOf(
            WelcomeSlide(
                icon = Icons.Rounded.WavingHand,
                title = "Welcome to Notes",
                body = "A simple workspace for folders, notes, and attachments on your device.",
            ),
            WelcomeSlide(
                icon = Icons.Rounded.Folder,
                title = "Organize with directories",
                body = "Group notes into folders. Use All Notes and Favorites for quick access.",
            ),
            WelcomeSlide(
                icon = Icons.Rounded.Note,
                title = "Write and attach",
                body = "Open a note to edit text, add images, and mark favorites.",
            ),
            WelcomeSlide(
                icon = Icons.Rounded.Cloud,
                title = "Ready to explore",
                body = "Next we will walk you through the main controls in the app.",
            ),
        )
    val pagerState = rememberPagerState(pageCount = { slides.size })
    val scope = rememberCoroutineScope()
    val colors = MaterialTheme.colorScheme
    val isLast = pagerState.currentPage == slides.lastIndex

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = colors.background,
    ) { padding ->
        Column(
            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(horizontal = 24.dp, vertical = 16.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.End,
            ) {
                TextButton(onClick = onSkip) {
                    Text("Skip")
                }
            }
            HorizontalPager(
                state = pagerState,
                modifier =
                    Modifier
                        .weight(1f)
                        .fillMaxWidth(),
            ) { page ->
                welcomeSlideContent(slide = slides[page])
            }
            welcomePagerIndicator(
                pagerState = pagerState,
                pageCount = slides.size,
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(vertical = 20.dp),
            )
            Button(
                onClick = {
                    if (isLast) {
                        onFinished()
                    } else {
                        scope.launch {
                            pagerState.animateScrollToPage(pagerState.currentPage + 1)
                        }
                    }
                },
                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(48.dp),
            ) {
                Text(if (isLast) "Get started" else "Next")
            }
            Spacer(Modifier.height(8.dp))
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun welcomePagerIndicator(
    pagerState: PagerState,
    pageCount: Int,
    modifier: Modifier = Modifier,
    activeWidth: Dp = 28.dp,
    dotSize: Dp = 8.dp,
    dotSpacing: Dp = 10.dp,
) {
    val colors = MaterialTheme.colorScheme
    val activeColor = colors.primary
    val inactiveColor = colors.onSurfaceVariant.copy(alpha = 0.35f)
    val scrollPosition by remember {
        derivedStateOf {
            pagerState.currentPage + pagerState.currentPageOffsetFraction
        }
    }

    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(dotSpacing, Alignment.CenterHorizontally),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        repeat(pageCount) { index ->
            val distance = (scrollPosition - index).absoluteValue.coerceIn(0f, 1f)
            val width = lerp(activeWidth, dotSize, distance)
            val color = lerp(activeColor, inactiveColor, distance)
            Box(
                modifier =
                    Modifier
                        .height(dotSize)
                        .width(width)
                        .clip(CircleShape)
                        .background(color),
            )
        }
    }
}

@Composable
private fun welcomeSlideContent(slide: WelcomeSlide) {
    val colors = MaterialTheme.colorScheme
    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .padding(horizontal = 8.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            imageVector = slide.icon,
            contentDescription = null,
            tint = colors.primary,
            modifier = Modifier.size(72.dp),
        )
        Spacer(Modifier.height(28.dp))
        Text(
            text = slide.title,
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.SemiBold,
            color = colors.onSurface,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(12.dp))
        Text(
            text = slide.body,
            style = MaterialTheme.typography.bodyLarge,
            color = colors.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )
    }
}
