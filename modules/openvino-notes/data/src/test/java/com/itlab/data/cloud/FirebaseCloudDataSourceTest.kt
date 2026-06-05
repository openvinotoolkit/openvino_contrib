package com.itlab.data.cloud

import com.google.android.gms.tasks.Task
import com.google.firebase.FirebaseException
import com.google.firebase.storage.FileDownloadTask
import com.google.firebase.storage.FirebaseStorage
import com.google.firebase.storage.ListResult
import com.google.firebase.storage.StorageMetadata
import com.google.firebase.storage.StorageReference
import com.google.firebase.storage.UploadTask
import com.itlab.domain.cloud.Result
import io.mockk.MockKAnnotations
import io.mockk.coEvery
import io.mockk.every
import io.mockk.impl.annotations.MockK
import io.mockk.mockk
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import io.mockk.verify
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.tasks.await
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.io.IOException

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class FirebaseCloudDataSourceTest {
    @MockK
    lateinit var storage: FirebaseStorage

    @MockK
    lateinit var rootRef: StorageReference

    @MockK
    lateinit var childRef: StorageReference

    private lateinit var dataSource: FirebaseCloudDataSource

    @Before
    fun setUp() {
        MockKAnnotations.init(this)

        mockkStatic("kotlinx.coroutines.tasks.TasksKt")

        every { storage.reference } returns rootRef
        dataSource = FirebaseCloudDataSource(storage)
    }

    @After
    fun tearDown() {
        unmockkAll()
    }

    @Test
    fun `listNoteMetadata success`() =
        runBlocking {
            val userId = "user123"
            val listResult = mockk<ListResult>()
            val itemRef = mockk<StorageReference>()
            val metadata = mockk<StorageMetadata>()
            val taskList = mockk<Task<ListResult>>()
            val taskMetadata = mockk<Task<StorageMetadata>>()

            every { rootRef.child("users/$userId/notes") } returns childRef
            every { childRef.listAll() } returns taskList

            coEvery { taskList.await() } returns listResult
            every { listResult.items } returns listOf(itemRef)
            every { itemRef.path } returns "notes/note1.json"
            every { itemRef.metadata } returns taskMetadata
            coEvery { taskMetadata.await() } returns metadata
            every { metadata.updatedTimeMillis } returns 1672531200000L // 2023-01-01

            val result = dataSource.listNoteMetadata(userId)

            assertTrue(result is Result.Success)
            val data = (result as Result.Success).data
            assertEquals(1, data.size)
            assertEquals("notes/note1.json", data[0].key)
        }

    @Test
    fun `listMediaMetadata success`() =
        runBlocking {
            val userId = "user1"
            val listResult = mockk<ListResult>()
            val itemRef = mockk<StorageReference>()
            val metadata = mockk<StorageMetadata>()

            val taskList = mockk<Task<ListResult>>()
            val taskMeta = mockk<Task<StorageMetadata>>()

            every { rootRef.child("users/$userId/media") } returns childRef
            every { childRef.listAll() } returns taskList
            coEvery { taskList.await() } returns listResult

            every { listResult.items } returns listOf(itemRef)
            every { itemRef.path } returns "users/user1/media/note1_id1"
            every { itemRef.name } returns "note1_id1"
            every { itemRef.metadata } returns taskMeta

            coEvery { taskMeta.await() } returns metadata
            every { metadata.contentType } returns "image/png"

            val result = dataSource.listMediaMetadata(userId)

            assertTrue(result is Result.Success)
            val data = (result as Result.Success).data
            assertEquals("note1_id1", data[0].mediaId)
            assertEquals("image/png", data[0].mimeType)
        }

    @Test
    fun `downloadNote success`() =
        runBlocking {
            val key = "note_key"
            val bytes = "note content".toByteArray()
            val task = mockk<Task<ByteArray>>()

            every { rootRef.child(key) } returns childRef
            every { childRef.getBytes(any()) } returns task
            coEvery { task.await() } returns bytes

            val result = dataSource.downloadNote(key)

            assertTrue(result is Result.Success)
            assertEquals("note content", (result as Result.Success).data)
        }

    @Test
    fun `uploadNote success`() =
        runBlocking {
            val task = mockk<UploadTask>()
            every { rootRef.child(any()) } returns childRef
            every { childRef.putBytes(any(), any()) } returns task
            coEvery { task.await() } returns mockk()

            val result = dataSource.uploadNote("key", "{}")

            assertTrue(result is Result.Success)
        }

    @Test
    fun `delete success`() =
        runBlocking {
            val task = mockk<Task<Void>>()
            every { rootRef.child(any()) } returns childRef
            every { childRef.delete() } returns task
            coEvery { task.await() } returns mockk()

            val result = dataSource.deleteObject("key")

            assertTrue(result is Result.Success)
        }

    @Test
    fun `uploadMedia success`() {
        runBlocking {
            val file = File.createTempFile("test", ".jpg")
            file.writeBytes(byteArrayOf(1, 2, 3))
            val task = mockk<UploadTask>()
            val mimeType = "image/jpeg"

            every { rootRef.child(any()) } returns childRef
            every { childRef.putFile(any<android.net.Uri>(), any()) } returns task
            coEvery { task.await() } returns mockk()

            val result =
                dataSource.uploadMedia(
                    "key",
                    com.itlab.domain.cloud
                        .DomainFile(file.absolutePath),
                    mimeType,
                )

            assertTrue(result is Result.Success)
            file.delete()
        }
    }

    @Test
    fun `downloadMedia success`() =
        runBlocking {
            val testFile = File("dummy/path/to/file")
            val task = mockk<FileDownloadTask>()

            every { rootRef.child(any()) } returns childRef
            every { childRef.getFile(any<File>()) } returns task
            coEvery { task.await() } returns mockk()

            val result =
                dataSource.downloadMedia(
                    "key",
                    com.itlab.domain.cloud
                        .DomainFile(testFile.absolutePath),
                )

            assertTrue(result is Result.Success)
        }

    @Test
    fun `safeCall catches FirebaseException`() =
        runBlocking {
            val exception = mockk<FirebaseException>()
            every { rootRef.child(any()) } throws exception

            val result = dataSource.deleteObject("key")

            assertTrue(result is Result.Error)
            assertEquals(exception, (result as Result.Error).exception)
        }

    @Test
    fun `safeCall catches IOException`() =
        runBlocking {
            val exception = IOException("Disk error")
            every { rootRef.child(any()) } throws exception

            val result = dataSource.deleteObject("key")

            assertTrue(result is Result.Error)
        }

    @Test
    fun `safeCall catches generic Exception`() =
        runBlocking {
            every { rootRef.child(any()) } throws RuntimeException("Boom")

            val result = dataSource.deleteObject("key")

            assertTrue(result is Result.Error)
        }

    @Test(expected = CancellationException::class)
    fun `safeCall rethrows CancellationException`() {
        runBlocking {
            every { rootRef.child(any()) } throws CancellationException("Cancelled")
            dataSource.deleteObject("key")
        }
    }

    @Test
    fun `deleteMedia success`() =
        runBlocking {
            val key = "media/photo.jpg"
            val task = mockk<Task<Void>>()

            every { rootRef.child(key) } returns childRef
            every { childRef.delete() } returns task

            coEvery { task.await() } returns mockk()

            val result = dataSource.deleteObject(key)

            assertTrue(result is Result.Success)
            verify { childRef.delete() }
        }

    @Test
    fun `deleteMedia failure`() =
        runBlocking {
            val key = "media/photo.jpg"
            val exception = RuntimeException("Firebase error")

            every { rootRef.child(key) } returns childRef
            every { childRef.delete() } throws exception

            val result = dataSource.deleteObject(key)

            assertTrue(result is Result.Error)
            assertEquals(exception, (result as Result.Error).exception)
        }

    @Test
    fun `listFolderMetadata success`() =
        runBlocking {
            val userId = "user123"
            val listResult = mockk<ListResult>()
            val itemRef = mockk<StorageReference>()
            val metadata = mockk<StorageMetadata>()
            val taskList = mockk<Task<ListResult>>()
            val taskMetadata = mockk<Task<StorageMetadata>>()

            every { rootRef.child("users/$userId/folders") } returns childRef
            every { childRef.listAll() } returns taskList

            coEvery { taskList.await() } returns listResult
            every { listResult.items } returns listOf(itemRef)
            every { itemRef.path } returns "users/$userId/folders/folder1.json"
            every { itemRef.metadata } returns taskMetadata
            coEvery { taskMetadata.await() } returns metadata
            every { metadata.updatedTimeMillis } returns 1672531200000L

            val result = dataSource.listFolderMetadata(userId)

            assertTrue(result is Result.Success)
            val data = (result as Result.Success).data
            assertEquals(1, data.size)
            assertEquals("users/$userId/folders/folder1.json", data[0].key)
        }

    @Test
    fun `downloadFolder success`() =
        runBlocking {
            val key = "users/user123/folders/folder1.json"
            val bytes = "{\"id\":\"1\",\"name\":\"Work\"}".toByteArray()
            val task = mockk<Task<ByteArray>>()

            every { rootRef.child(key) } returns childRef
            every { childRef.getBytes(any()) } returns task
            coEvery { task.await() } returns bytes

            val result = dataSource.downloadFolder(key)

            assertTrue(result is Result.Success)
            assertEquals("{\"id\":\"1\",\"name\":\"Work\"}", (result as Result.Success).data)
        }

    @Test
    fun `uploadFolder success`() =
        runBlocking {
            val key = "users/user123/folders/folder1.json"
            val json = "{\"id\":\"1\",\"name\":\"Work\"}"
            val task = mockk<UploadTask>()

            every { rootRef.child(key) } returns childRef
            every { childRef.putBytes(any(), any()) } returns task
            coEvery { task.await() } returns mockk()

            val result = dataSource.uploadFolder(key, json)

            assertTrue(result is Result.Success)
            verify { childRef.putBytes(any(), any()) }
        }

    @Test
    fun `deleteFolder success`() =
        runBlocking {
            val key = "users/user123/folders/folder1.json"
            val task = mockk<Task<Void>>()

            every { rootRef.child(key) } returns childRef
            every { childRef.delete() } returns task
            coEvery { task.await() } returns mockk()

            val result = dataSource.deleteObject(key)

            assertTrue(result is Result.Success)
            verify { childRef.delete() }
        }

    @Test
    fun `deleteFolder failure`() =
        runBlocking {
            val key = "users/user123/folders/folder1.json"
            val exception = RuntimeException("Firebase storage error")

            every { rootRef.child(key) } returns childRef
            every { childRef.delete() } throws exception

            val result = dataSource.deleteObject(key)

            assertTrue(result is Result.Error)
            assertEquals(exception, (result as Result.Error).exception)
        }
}
