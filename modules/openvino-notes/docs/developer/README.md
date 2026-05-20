# Developer Guide

This documentation is intended for contributors working on `openvino-notes`.

The repository already has a meaningful CI and build setup, while the application code is still at an early implementation stage. The goal of these documents is to help contributors understand the project quickly and reproduce the same checks that gate pull requests and `main`.

## Recommended Reading Order

1. [Local CI Reproduction](./ci-local.md)
2. [Project Overview](./project.md)

## Current State

What is already in place:

- a four-module Android build
- reusable GitHub Actions workflows
- shared formatting, lint, and coverage policy

What is still mostly scaffolded:

- domain contracts
- data-layer behavior
- OpenVINO integration
- app-level product flows

## Main Work Areas

- Application code: `app`, `domain`, `data`, `ai`
- Automation and CI: `.github`

# Domain Layer Documentation

This documentation describes the Domain layer of the `openvino-notes` application. It is intended for contributors working on business logic, AI integration, and unit testing.

The Domain layer is the central part of the architecture, defining business rules for notes and folders, AI operations, and repository contracts. It is independent of storage, UI, and AI implementation details.

## Purpose

- Maintain business logic separately from UI, storage, and AI.
- Define domain entities, repository contracts, and use cases.
- Provide testable interfaces for both normal and AI-enhanced operations.

## Layer Responsibilities

The Domain layer contains:

- Domain models (`Note`, `NoteFolder`)
- Repository interfaces (`NotesRepository`, `NoteFolderRepository`)
- Use cases for notes and folders
- AI service interface (`NoteAiService`)
- AI-related use cases (`SuggestSummaryUseCase`, `ApplyTagsUseCase`, etc.)

The Domain layer **does not contain**:

- UI elements or ViewModels
- Android-specific code
- Implementation details of repositories or AI
- OpenVINO or network code

## Structure
domain/
├─ ai/
│ └─ NoteAiService.kt
├─ aiusecase/
│ ├─ ApplySummaryUseCase.kt
│ ├─ ApplyTagsUseCase.kt
│ ├─ SuggestSummaryUseCase.kt
│ └─ SuggestTagsUseCase.kt
├─ model/
│ ├─ Note.kt
│ ├─ NoteFolder.kt
│ └─ ContentItem.kt
├─ repository/
│ ├─ NotesRepository.kt
│ └─ NoteFolderRepository.kt
└─ usecase/
├─ CreateNoteUseCase.kt
├─ DeleteNoteUseCase.kt
├─ GetNoteUseCase.kt
├─ UpdateNoteUseCase.kt
└─ MoveNoteToFolderUseCase.kt


## Domain Models

### Note

- `id`: unique identifier
- `title`: note title
- `folderId`: optional folder ID
- `contentItems`: list of `ContentItem`
- `createdAt`: creation timestamp
- `updatedAt`: last update timestamp
- `tags`: set of tags
- `isFavorite`: favorite flag
- `summary`: optional AI-generated summary

### ContentItem

Sealed class representing a note's content:

- `Text`: text block
- `Image`: image block (`Local` or `Remote` source)
- `File`: file attachment
- `Link`: URL link

### TextFormat

- `PLAIN`, `MARKDOWN`, `HTML`

### ImageSource

- `Local`: local file path
- `Remote`: remote URL

### NoteFolder

- `id`: unique identifier
- `name`: folder name
- `createdAt`, `updatedAt`: timestamps
- `metadata`: optional extra info

## Repository Interfaces

### NotesRepository

- `observeNotes()`: flow of all notes
- `observeNotesByFolder(folderId)`: flow of notes for a folder
- `getNoteById(id)`: retrieve note by ID
- `createNote(note)`: create note
- `updateNote(note)`: update note
- `deleteNote(id)`: delete note

### NoteFolderRepository

- `observeFolders()`: flow of folders
- `createFolder(folder)`: create folder
- `renameFolder(id, name)`: rename folder
- `deleteFolder(id)`: delete folder
- `getFolderById(id)`: retrieve folder by ID
- `updateFolder(folder)`: update folder

Repositories abstract storage for testable domain logic.

## Use Cases

### Folder Use Cases

- `CreateFolderUseCase`
- `DeleteFolderUseCase`
- `GetFolderUseCase`
- `ObserveFoldersUseCase`
- `UpdateFolderUseCase`

### Note Use Cases

- `CreateNoteUseCase`
- `DeleteNoteUseCase`
- `GetNoteUseCase`
- `UpdateNoteUseCase`
- `MoveNoteToFolderUseCase`
- `ObserveNotesUseCase`
- `ObserveNotesByFolderUseCase`

## AI Contract

### NoteAiService

Interface for AI operations.

- `suspend fun summarize(text: String): String`
- `suspend fun tagTXT(text: String): Set<String>`
- `suspend fun tagIMGs(images: List<String>): Set<String>`

## AI Use Cases

- `SuggestSummaryUseCase`: extract text, call AI, return proposed summary
- `SuggestTagsUseCase`: extract text/images, call AI, return combined tags
- `ApplySummaryUseCase`: update note with AI-generated summary
- `ApplyTagsUseCase`: update note with AI-generated tags

AI operations are separated into **suggest** (proposal) and **apply** (commit) stages.

## Data Flow

### Normal Note Operations

1. UI triggers an action
2. ViewModel calls a use case
3. Use case interacts with repository
4. Repository returns or saves domain models
5. Result propagates back to ViewModel
6. ViewModel updates UI

### AI Operations

1. UI triggers AI action
2. ViewModel calls `SuggestSummaryUseCase` or `SuggestTagsUseCase`
3. Use case retrieves note from repository
4. Use case extracts content
5. Use case calls `NoteAiService`
6. AI returns results
7. ViewModel receives results
8. If confirmed, `ApplySummaryUseCase` or `ApplyTagsUseCase` updates the note

## Principles

- Android-agnostic
- Storage-agnostic
- AI-implementation-agnostic
- “Suggest” and “Apply” AI operations are separated
- Models are extensible

## Testing

Unit tests cover:

- Creating, updating, deleting notes and folders
- Moving notes between folders
- Observing notes and folders
- Getting AI-generated summaries and tags
- Applying summaries and tags
- Error handling for missing notes

Fake repositories and fake AI service enable testing without Android or OpenVINO dependencies.