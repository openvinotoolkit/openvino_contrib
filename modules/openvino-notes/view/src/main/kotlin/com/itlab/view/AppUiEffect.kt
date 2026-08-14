package com.itlab.view

sealed interface AppUiEffect {
    data class Message(val text: String) : AppUiEffect
    data object OpenAccountSettings : AppUiEffect
}
