package com.itlab.domain.usecase

internal fun requireNotBlank(
    value: String,
    fieldName: String,
) {
    require(value.isNotBlank()) { "$fieldName must not be blank" }
}
