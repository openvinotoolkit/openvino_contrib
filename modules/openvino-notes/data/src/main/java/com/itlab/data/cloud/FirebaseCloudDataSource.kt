package com.itlab.data.cloud

import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.StorageReference
import com.itlab.domain.cloud.CloudDataSource
import com.itlab.domain.cloud.CloudMediaMetadata
import com.itlab.domain.cloud.CloudMetadata
import com.itlab.domain.cloud.DomainFile
import com.itlab.domain.cloud.Result
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.tasks.await
import java.io.File
import kotlin.time.Instant

private const val MAX_NOTE_SIZE = 5 * 1024 * 1024L

class FirebaseCloudDataSource(
    private val storage: FirebaseStorage = FirebaseStorage.getInstance(),
) : CloudDataSource {
    private val rootRef = storage.reference

    private val noteDispatcher = FirebaseNoteDispatcher(rootRef)
    private val folderDispatcher = FirebaseFolderDispatcher(rootRef)
    private val mediaDispatcher = FirebaseMediaDispatcher(rootRef)

    override suspend fun listNoteMetadata(userId: String): Result<List<CloudMetadata>> =
        safeCall {
            noteDispatcher.list(userId)
        }

    override suspend fun downloadNote(key: String): Result<String> = safeCall { noteDispatcher.download(key) }

    override suspend fun uploadNote(
        key: String,
        json: String,
    ): Result<Unit> = safeCall { noteDispatcher.upload(key, json) }

    override suspend fun listFolderMetadata(userId: String): Result<List<CloudMetadata>> =
        safeCall {
            folderDispatcher.list(userId)
        }

    override suspend fun downloadFolder(key: String): Result<String> = safeCall { folderDispatcher.download(key) }

    override suspend fun uploadFolder(
        key: String,
        json: String,
    ): Result<Unit> =
        safeCall {
            folderDispatcher.upload(key, json)
        }

    override suspend fun listMediaMetadata(userId: String): Result<List<CloudMediaMetadata>> =
        safeCall {
            mediaDispatcher.list(userId)
        }

    override suspend fun uploadMedia(
        key: String,
        file: DomainFile,
        mimeType: String,
    ): Result<Unit> =
        safeCall {
            mediaDispatcher.upload(key, file, mimeType)
        }

    override suspend fun downloadMedia(
        key: String,
        destination: DomainFile,
    ): Result<Unit> =
        safeCall {
            mediaDispatcher.download(key, destination)
        }

    override suspend fun deleteObject(key: String): Result<Unit> = safeCall { deleteRemoteFile(rootRef, key) }

    private class FirebaseNoteDispatcher(
        private val rootRef: StorageReference,
    ) {
        suspend fun list(userId: String): List<CloudMetadata> = listMetadataHelper(rootRef, "users/$userId/notes")

        suspend fun download(key: String): String = downloadJsonHelper(rootRef, key)

        suspend fun upload(
            key: String,
            json: String,
        ) = uploadJsonHelper(rootRef, key, json)
    }

    private class FirebaseFolderDispatcher(
        private val rootRef: StorageReference,
    ) {
        suspend fun list(userId: String): List<CloudMetadata> = listMetadataHelper(rootRef, "users/$userId/folders")

        suspend fun download(key: String): String = downloadJsonHelper(rootRef, key)

        suspend fun upload(
            key: String,
            json: String,
        ) = uploadJsonHelper(rootRef, key, json)
    }

    private class FirebaseMediaDispatcher(
        private val rootRef: StorageReference,
    ) {
        suspend fun list(userId: String): List<CloudMediaMetadata> {
            val mediaRef = rootRef.child("users/$userId/media")
            val result = mediaRef.listAll().await()
            return coroutineScope {
                result.items
                    .map { itemRef ->
                        async {
                            val metadata = itemRef.metadata.await()
                            CloudMediaMetadata(
                                key = itemRef.path,
                                mediaId = itemRef.name,
                                mimeType = metadata.contentType ?: "application/octet-stream",
                            )
                        }
                    }.awaitAll()
            }
        }

        suspend fun upload(
            key: String,
            file: DomainFile,
            mimeType: String,
        ) {
            val localFile = File(file.path)
            require(localFile.isFile && localFile.length() > 0L) {
                "Media file is missing or empty: ${file.path}"
            }
            val metadata =
                com.google.firebase.storage
                    .storageMetadata { contentType = mimeType }
            rootRef
                .child(key)
                .putFile(android.net.Uri.fromFile(localFile), metadata)
                .await()
        }

        suspend fun download(
            key: String,
            destination: DomainFile,
        ) {
            rootRef.child(key).getFile(File(destination.path)).await()
        }
    }
}

internal suspend fun listMetadataHelper(
    rootRef: StorageReference,
    path: String,
): List<CloudMetadata> {
    val listRef = rootRef.child(path)
    val result = listRef.listAll().await()
    return coroutineScope {
        result.items
            .map { itemRef ->
                async {
                    val metadata = itemRef.metadata.await()
                    CloudMetadata(
                        key = itemRef.path,
                        updatedAt = Instant.fromEpochMilliseconds(metadata.updatedTimeMillis),
                    )
                }
            }.awaitAll()
    }
}

internal suspend fun downloadJsonHelper(
    rootRef: StorageReference,
    key: String,
): String {
    val bytes = rootRef.child(key).getBytes(MAX_NOTE_SIZE).await()
    return String(bytes)
}

internal suspend fun uploadJsonHelper(
    rootRef: StorageReference,
    key: String,
    json: String,
) {
    val metadata =
        com.google.firebase.storage
            .storageMetadata { contentType = "application/json" }
    rootRef.child(key).putBytes(json.toByteArray(), metadata).await()
}

internal suspend fun deleteRemoteFile(
    rootRef: StorageReference,
    key: String,
) {
    rootRef.child(key).delete().await()
}

@Suppress("TooGenericExceptionCaught")
internal suspend inline fun <T> safeCall(crossinline block: suspend () -> T): Result<T> =
    try {
        Result.Success(block())
    } catch (e: CancellationException) {
        throw e
    } catch (e: com.google.firebase.FirebaseException) {
        Result.Error(e)
    } catch (e: java.io.IOException) {
        Result.Error(e)
    } catch (e: Exception) {
        Result.Error(e)
    }
