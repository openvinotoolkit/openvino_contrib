package com.itlab.data.cloud

/**
 * Storage object name: `{noteId}_{mediaId}`.
 * Uses the last underscore so UUID note/media ids stay intact.
 */
internal object MediaStorageNaming {
    fun compose(
        noteId: String,
        mediaId: String,
    ): String = "${noteId}_$mediaId"

    fun parse(storageName: String): Pair<String, String> {
        val index = storageName.lastIndexOf('_')
        require(index > 0 && index < storageName.lastIndex) {
            "Invalid media storage name: $storageName"
        }
        return storageName.substring(0, index) to storageName.substring(index + 1)
    }
}
