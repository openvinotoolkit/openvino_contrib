// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.notes.room

import com.openvino.notes.kernel.AccountKey
import com.openvino.notes.kernel.AppDispatchers
import com.openvino.notes.notes.api.AttachmentId
import com.openvino.notes.notes.api.AttachmentMetadata
import com.openvino.notes.notes.api.port.AttachmentContentConflictException
import com.openvino.notes.notes.api.port.AttachmentContentPort
import com.openvino.notes.notes.api.port.BinarySource
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

internal class FileAttachmentContentStore(
    private val root: File,
    private val dispatchers: AppDispatchers,
) : AttachmentContentPort {
    override suspend fun put(
        accountKey: AccountKey,
        attachment: AttachmentMetadata,
        source: BinarySource,
    ) = withContext(dispatchers.io) {
        require(source.sizeBytes >= 0) { "Attachment source size must not be negative" }
        require(source.sizeBytes == attachment.sizeBytes) {
            "Attachment content size does not match metadata"
        }
        val target = contentFile(accountKey, attachment.id)
        val directory = requireNotNull(target.parentFile)
        check(directory.isDirectory || directory.mkdirs()) { "Unable to create attachment content directory" }
        val temporary = File.createTempFile("attachment-", ".tmp", directory)
        try {
            val digest = MessageDigest.getInstance(CONTENT_DIGEST_ALGORITHM)
            temporary.outputStream().buffered().use { output ->
                var offset = 0L
                while (offset < source.sizeBytes) {
                    currentCoroutineContext().ensureActive()
                    val requested = minOf(COPY_CHUNK_BYTES.toLong(), source.sizeBytes - offset).toInt()
                    val chunk = source.read(offset, requested)
                    check(chunk.isNotEmpty()) { "Attachment source ended before its declared size" }
                    check(chunk.size <= requested) { "Attachment source returned more bytes than requested" }
                    output.write(chunk)
                    digest.update(chunk)
                    offset += chunk.size
                }
            }
            RandomAccessFile(directory.resolve(CONTENT_LOCK_FILE), "rw").channel.use { channel ->
                channel.lock().use {
                    if (target.isFile) {
                        requireSameContent(target, source.sizeBytes, digest.digest(), attachment.id)
                    } else {
                        try {
                            Files.move(temporary.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE)
                        } catch (_: AtomicMoveNotSupportedException) {
                            Files.move(temporary.toPath(), target.toPath())
                        }
                    }
                }
            }
        } finally {
            temporary.delete()
        }
        Unit
    }

    override suspend fun open(accountKey: AccountKey, attachmentId: AttachmentId): BinarySource? =
        withContext(dispatchers.io) {
            val file = contentFile(accountKey, attachmentId)
            if (file.isFile) FileBinarySource(file, dispatchers) else null
        }

    override suspend fun delete(accountKey: AccountKey, attachmentId: AttachmentId): Boolean =
        withContext(dispatchers.io) { contentFile(accountKey, attachmentId).delete() }

    private fun contentFile(accountKey: AccountKey, attachmentId: AttachmentId): File =
        root.resolve(accountKey.value.sha256()).resolve("${attachmentId.value.sha256()}.bin")

    private suspend fun requireSameContent(
        existing: File,
        expectedSize: Long,
        expectedDigest: ByteArray,
        attachmentId: AttachmentId,
    ) {
        val sameSize = existing.length() == expectedSize
        val sameDigest = sameSize && MessageDigest.isEqual(existing.sha256Bytes(), expectedDigest)
        if (!sameDigest) throw AttachmentContentConflictException(attachmentId)
    }
}

private class FileBinarySource(
    private val file: File,
    private val dispatchers: AppDispatchers,
    override val sizeBytes: Long = file.length(),
) : BinarySource {
    override suspend fun read(offset: Long, maxBytes: Int): ByteArray = withContext(dispatchers.io) {
        require(offset >= 0) { "offset must not be negative" }
        require(maxBytes > 0) { "maxBytes must be positive" }
        currentCoroutineContext().ensureActive()
        if (offset >= sizeBytes) return@withContext byteArrayOf()
        val requested = minOf(maxBytes.toLong(), sizeBytes - offset).toInt()
        RandomAccessFile(file, "r").use { input ->
            input.seek(offset)
            val buffer = ByteArray(requested)
            val count = input.read(buffer)
            check(count > 0) { "Attachment file ended before its declared size" }
            if (count == buffer.size) buffer else buffer.copyOf(count)
        }
    }
}

private fun String.sha256(): String = MessageDigest.getInstance("SHA-256")
    .digest(toByteArray(Charsets.UTF_8))
    .joinToString("") { byte -> "%02x".format(byte) }

private suspend fun File.sha256Bytes(): ByteArray {
    val digest = MessageDigest.getInstance(CONTENT_DIGEST_ALGORITHM)
    inputStream().buffered().use { input ->
        val buffer = ByteArray(COPY_CHUNK_BYTES)
        while (true) {
            currentCoroutineContext().ensureActive()
            val count = input.read(buffer)
            if (count < 0) break
            digest.update(buffer, 0, count)
        }
    }
    return digest.digest()
}

private const val COPY_CHUNK_BYTES = 64 * 1024
private const val CONTENT_DIGEST_ALGORITHM = "SHA-256"
private const val CONTENT_LOCK_FILE = ".attachment-content.lock"
