package com.itlab.data.db

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import com.itlab.data.dao.FolderDao
import com.itlab.data.dao.MediaDao
import com.itlab.data.dao.NoteDao
import com.itlab.data.entity.FolderEntity
import com.itlab.data.entity.MediaEntity
import com.itlab.data.entity.NoteEntity

@Database(
    entities = [NoteEntity::class, MediaEntity::class, FolderEntity::class],
    version = 1,
    exportSchema = false,
)
@TypeConverters(DateTimeConverters::class, MetadataConverter::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun noteDao(): NoteDao

    abstract fun mediaDao(): MediaDao

    abstract fun folderDao(): FolderDao
}
