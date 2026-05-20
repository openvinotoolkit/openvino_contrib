package com.itlab.notes.ui.notes

import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp

@Composable
fun appSearchField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    placeholderText: String = "Search",
) {
    var isFocused by remember { mutableStateOf(false) }
    val focusManager = LocalFocusManager.current
    val interactionSource = remember { MutableInteractionSource() }
    val onClearClick: () -> Unit = {
        onValueChange("")
        isFocused = false
        focusManager.clearFocus(force = true)
    }

    Box(
        modifier =
            modifier
                .fillMaxWidth()
                .height(56.dp),
        contentAlignment = Alignment.Center,
    ) {
        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            modifier =
                Modifier
                    .fillMaxWidth()
                    .height(48.dp)
                    .onFocusChanged { isFocused = it.isFocused },
            singleLine = true,
            interactionSource = interactionSource,
            textStyle = MaterialTheme.typography.bodyLarge.copy(color = MaterialTheme.colorScheme.onSurface),
            cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
            decorationBox = { content ->
                appSearchFieldDecorationBox(
                    input =
                        AppSearchFieldDecorationInput(
                            value = value,
                            isFocused = isFocused,
                            placeholderText = placeholderText,
                            onClearClick = onClearClick,
                        ),
                    interactionSource = interactionSource,
                ) {
                    content()
                }
            },
        )
    }
}

@Composable
private fun appSearchFieldDecorationBox(
    input: AppSearchFieldDecorationInput,
    interactionSource: MutableInteractionSource,
    content: @Composable () -> Unit,
) {
    TextFieldDefaults.DecorationBox(
        value = input.value,
        innerTextField = content,
        enabled = true,
        singleLine = true,
        visualTransformation = VisualTransformation.None,
        interactionSource = interactionSource,
        placeholder = {
            Text(
                input.placeholderText,
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        },
        leadingIcon = {
            appSearchFieldLeadingIcon(isFocused = input.isFocused)
        },
        trailingIcon = {
            appSearchFieldTrailingIcon(
                isFocused = input.isFocused,
                onClearClick = input.onClearClick,
            )
        },
        shape = CircleShape,
        colors =
            TextFieldDefaults.colors(
                focusedIndicatorColor = Color.Transparent,
                unfocusedIndicatorColor = Color.Transparent,
                disabledIndicatorColor = Color.Transparent,
                focusedContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainer,
                cursorColor = MaterialTheme.colorScheme.primary,
            ),
        contentPadding = PaddingValues(start = 24.dp, end = 16.dp, top = 4.dp, bottom = 4.dp),
    )
}

@Composable
private fun appSearchFieldLeadingIcon(isFocused: Boolean) {
    Icon(
        imageVector = Icons.Rounded.Search,
        contentDescription = null,
        tint =
            if (isFocused) {
                MaterialTheme.colorScheme.primary
            } else {
                MaterialTheme.colorScheme.onSurfaceVariant
            },
        modifier =
            Modifier
                .padding(start = 24.dp)
                .size(25.dp),
    )
}

@Composable
private fun appSearchFieldTrailingIcon(
    isFocused: Boolean,
    onClearClick: () -> Unit,
) {
    if (isFocused) {
        IconButton(onClick = onClearClick) {
            Icon(
                imageVector = Icons.Rounded.Close,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private data class AppSearchFieldDecorationInput(
    val value: String,
    val isFocused: Boolean,
    val placeholderText: String,
    val onClearClick: () -> Unit,
)
