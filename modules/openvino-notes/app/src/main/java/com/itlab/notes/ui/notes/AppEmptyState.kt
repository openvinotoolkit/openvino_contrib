package com.itlab.notes.ui.notes

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Folder
import androidx.compose.material.icons.rounded.SearchOff
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

@Composable
fun appEmptyState(
    icon: ImageVector,
    title: String,
    message: String,
    modifier: Modifier = Modifier,
) {
    val colors = MaterialTheme.colorScheme
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier.fillMaxWidth(),
    ) {
        Box(
            modifier =
                Modifier
                    .clip(MaterialTheme.shapes.medium)
                    .background(colors.surfaceContainer),
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.padding(14.dp).size(32.dp),
                tint = colors.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(16.dp))
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            color = colors.onSurface,
            textAlign = TextAlign.Center,
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = colors.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier.padding(horizontal = 50.dp),
        )
    }
}

@Composable
fun notesSearchEmptyState(modifier: Modifier = Modifier) {
    appEmptyState(
        icon = Icons.Rounded.SearchOff,
        title = "No results found",
        message = "Try a different search term or check another folder.",
        modifier = modifier,
    )
}

@Composable
fun directoriesSearchEmptyState(modifier: Modifier = Modifier) {
    appEmptyState(
        icon = Icons.Rounded.SearchOff,
        title = "No results found",
        message = "Try a different search term.",
        modifier = modifier,
    )
}

@Composable
fun directoriesEmptyState(modifier: Modifier = Modifier) {
    appEmptyState(
        icon = Icons.Rounded.Folder,
        title = "No directories yet",
        message = "Tap + to create a directory and organize your notes.",
        modifier = modifier,
    )
}
