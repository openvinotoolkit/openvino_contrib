package com.itlab.data.db

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.annotation.Config

@RunWith(AndroidJUnit4::class)
@Config(manifest = Config.NONE, sdk = [34])
class AppDatabaseTest {
    private lateinit var db: AppDatabase

    @Before
    fun createDb() {
        db =
            Room
                .inMemoryDatabaseBuilder(
                    ApplicationProvider.getApplicationContext(),
                    AppDatabase::class.java,
                ).allowMainThreadQueries()
                .build()
    }

    @After
    fun claseDb() {
        db.close()
    }

    @Test
    fun `database should provide all daos`() {
        assertNotNull(db.noteDao())
        assertNotNull(db.mediaDao())
    }
}
