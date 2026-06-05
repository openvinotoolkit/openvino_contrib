package com.itlab.notes.ui

private val LINE_BREAK_REGEX = Regex("[\r\n\u2028\u2029\u0085]+")

/** Removes line breaks (typing, paste, or legacy data) so text stays on one line. */
fun String.toSingleLineText(): String = replace(LINE_BREAK_REGEX, " ")
