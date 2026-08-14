// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

package com.itlab.app

import android.content.Context
import androidx.work.WorkerFactory
import com.itlab.ai.image.openvino.OpenVinoImageComponent
import com.itlab.ai.text.openvino.OpenVinoTextComponent
import com.itlab.assistant.api.NoteAssistant
import com.itlab.assistant.core.AssistantCoreComponent
import com.itlab.identity.api.IdentityService
import com.itlab.identity.google.GoogleIdentityComponent
import com.itlab.identity.google.GoogleIdentityUiController
import com.itlab.kernel.NoOpAppLogger
import com.itlab.kernel.SystemAppClock
import com.itlab.notes.api.NotesService
import com.itlab.notes.core.NotesCoreComponent
import com.itlab.notes.room.RoomNotesComponent
import com.itlab.settings.api.SettingsService
import com.itlab.settings.datastore.DataStoreSettingsComponent
import com.itlab.sync.android.AndroidSyncComponent
import com.itlab.sync.api.SyncScheduler
import com.itlab.sync.api.SyncService
import com.itlab.sync.core.SyncCoreComponent

class AppComposition(context: Context) : AutoCloseable {
    private val appContext = context.applicationContext
    private val identity = GoogleIdentityComponent.create()
    private val notesStorage = RoomNotesComponent.create(appContext)
    private val settings = DataStoreSettingsComponent.create(appContext)
    private val notesCore = NotesCoreComponent.create(identity.identityService, notesStorage.repository, SystemAppClock)
    private val textAi = OpenVinoTextComponent.create()
    private val imageAi = OpenVinoImageComponent.create()
    private val syncCore = SyncCoreComponent.create(NoOpAppLogger)
    private val assistantCore = AssistantCoreComponent.create(notesCore.notesService, textAi.assistant, imageAi.tagger)

    val identityService: IdentityService = identity.identityService
    val notesService: NotesService = notesCore.notesService
    val settingsService: SettingsService = settings.service
    val syncService: SyncService = syncCore.service
    val noteAssistant: NoteAssistant = assistantCore.assistant
    val workerFactory: WorkerFactory = AndroidSyncComponent.createWorkerFactory(syncCore.executor)
    val syncScheduler: SyncScheduler by lazy { AndroidSyncComponent.createScheduler(appContext) }

    fun createActivityIdentityController(): GoogleIdentityUiController = identity.createActivityController()

    override fun close() {
        textAi.assistant.close()
        imageAi.tagger.close()
        notesStorage.close()
    }
}
