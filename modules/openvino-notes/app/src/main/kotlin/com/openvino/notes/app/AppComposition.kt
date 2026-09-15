// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.openvino.notes.app

import android.content.Context
import android.app.Activity
import androidx.work.WorkerFactory
import com.openvino.notes.ai.image.openvino.OpenVinoImageComponent
import com.openvino.notes.ai.text.openvino.OpenVinoTextComponent
import com.openvino.notes.assistant.api.NoteAssistant
import com.openvino.notes.assistant.core.AssistantCoreComponent
import com.openvino.notes.identity.api.IdentityService
import com.openvino.notes.identity.google.GoogleIdentityComponent
import com.openvino.notes.identity.google.GoogleIdentityUiController
import com.openvino.notes.kernel.AppDispatchers
import com.openvino.notes.kernel.NoOpAppLogger
import com.openvino.notes.kernel.SystemAppClock
import com.openvino.notes.notes.api.NotesService
import com.openvino.notes.notes.api.FolderService
import com.openvino.notes.notes.core.NotesCoreComponent
import com.openvino.notes.notes.room.RoomNotesComponent
import com.openvino.notes.settings.api.SettingsService
import com.openvino.notes.settings.datastore.DataStoreSettingsComponent
import com.openvino.notes.sync.android.AndroidSyncComponent
import com.openvino.notes.sync.api.SyncService
import com.openvino.notes.sync.api.port.SyncScheduler
import com.openvino.notes.sync.core.SyncCoreComponent

class AppComposition(context: Context) : AutoCloseable {
    private val appContext = context.applicationContext
    private val dispatchers = AppDispatchers.production()
    private val identity = GoogleIdentityComponent.create()
    private val notesStorage = RoomNotesComponent.create(appContext, dispatchers = dispatchers)
    private val settings = DataStoreSettingsComponent.create(appContext)
    private val notesCore = NotesCoreComponent.create(
        identity.identityService,
        notesStorage.repository,
        notesStorage.folderRepository,
        SystemAppClock,
    )
    private val textAi = OpenVinoTextComponent.create()
    private val imageAi = OpenVinoImageComponent.create()
    private val syncStorage = AndroidSyncComponent.openPersistence(appContext)
    private val syncCore = SyncCoreComponent.create(
        NoOpAppLogger,
        syncStorage.checkpointPort,
        syncStorage.transferCheckpointPort,
    )
    private val assistantCore = AssistantCoreComponent.create(
        notesCore.notesService,
        notesStorage.attachmentContent,
        textAi.assistant,
        imageAi.tagger,
    )

    val identityService: IdentityService = identity.identityService
    val notesService: NotesService = notesCore.notesService
    val folderService: FolderService = notesCore.folderService
    val settingsService: SettingsService = settings.service
    val syncService: SyncService = syncCore.service
    val noteAssistant: NoteAssistant = assistantCore.assistant
    val workerFactory: WorkerFactory = AndroidSyncComponent.createWorkerFactory(syncCore.executor)
    val syncScheduler: SyncScheduler by lazy { AndroidSyncComponent.createScheduler(appContext) }

    fun createActivityIdentityController(activity: Activity): GoogleIdentityUiController =
        identity.createActivityController(activity)

    override fun close() {
        textAi.assistant.close()
        imageAi.tagger.close()
        syncStorage.close()
        notesStorage.close()
    }
}
