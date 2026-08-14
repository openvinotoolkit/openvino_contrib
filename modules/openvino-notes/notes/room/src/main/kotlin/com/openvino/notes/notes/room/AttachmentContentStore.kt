// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.notes.api.AttachmentContentPort
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.BinarySource
import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal class FileAttachmentContentStore(private val root: File) : AttachmentContentPort {
    override suspend fun put(
        accountKey: AccountKey,
        attachment: AttachmentMetadata,
        source: BinarySource,
    ) = withContext(Dispatchers.IO) {
        val bytes = source.read()
        require(bytes.size.toLong() == attachment.sizeBytes) {
            "Attachment content size does not match metadata"
        }
        val target = contentFile(accountKey, attachment.id)
        target.parentFile?.mkdirs()
        val temporary = File.createTempFile("attachment-", ".tmp", target.parentFile)
        try {
            temporary.outputStream().use { it.write(bytes) }
            try {
                Files.move(
                    temporary.toPath(),
                    target.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING,
                )
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally {
            temporary.delete()
        }
        Unit
    }

    override suspend fun open(accountKey: AccountKey, attachmentId: AttachmentId): BinarySource? =
        withContext(Dispatchers.IO) {
            val file = contentFile(accountKey, attachmentId)
            if (file.isFile) BinarySource { file.readBytes() } else null
        }

    override suspend fun delete(accountKey: AccountKey, attachmentId: AttachmentId): Boolean =
        withContext(Dispatchers.IO) { contentFile(accountKey, attachmentId).delete() }

    private fun contentFile(accountKey: AccountKey, attachmentId: AttachmentId): File =
        root.resolve(accountKey.value.sha256()).resolve("${attachmentId.value.sha256()}.bin")
}

private fun String.sha256(): String = MessageDigest.getInstance("SHA-256")
    .digest(toByteArray(Charsets.UTF_8))
    .joinToString("") { byte -> "%02x".format(byte) }
