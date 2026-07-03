package llm

import (
	parent "github.com/ollama/ollama/llm"
)

// Re-export types from parent llm package for backward compatibility
// This package exists so that code can import "github.com/ollama/ollama/llm/llama"
// as `llamaserver` to distinguish from the genai server package.

type LlamaServer = parent.LlamaServer
type ImageData = parent.ImageData
type CompletionRequest = parent.CompletionRequest
type CompletionResponse = parent.CompletionResponse
type DoneReason = parent.DoneReason
type ServerStatus = parent.ServerStatus
type ServerStatusResponse = parent.ServerStatusResponse
type LoadOperation = parent.LoadOperation
type LoadRequest = parent.LoadRequest
type LoadResponse = parent.LoadResponse
type EmbeddingRequest = parent.EmbeddingRequest
type EmbeddingResponse = parent.EmbeddingResponse
type StatusWriter = parent.StatusWriter
type TokenLogprob = parent.TokenLogprob
type Logprob = parent.Logprob

const (
	DoneReasonStop             = parent.DoneReasonStop
	DoneReasonLength           = parent.DoneReasonLength
	DoneReasonConnectionClosed = parent.DoneReasonConnectionClosed

	ServerStatusReady            = parent.ServerStatusReady
	ServerStatusNoSlotsAvailable = parent.ServerStatusNoSlotsAvailable
	ServerStatusLaunched         = parent.ServerStatusLaunched
	ServerStatusLoadingModel     = parent.ServerStatusLoadingModel
	ServerStatusNotResponding    = parent.ServerStatusNotResponding
	ServerStatusError            = parent.ServerStatusError

	LoadOperationFit    = parent.LoadOperationFit
	LoadOperationAlloc  = parent.LoadOperationAlloc
	LoadOperationCommit = parent.LoadOperationCommit
	LoadOperationClose  = parent.LoadOperationClose
)

var (
	NewLlamaServer      = parent.NewLlamaServer
	LoadModel           = parent.LoadModel
	ErrLoadRequiredFull = parent.ErrLoadRequiredFull
	NewStatusWriter     = parent.NewStatusWriter
)
