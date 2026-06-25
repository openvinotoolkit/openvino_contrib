package common

import (
	"strings"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/api"
	llamaserver "github.com/ollama/ollama/llm/llama"
)

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type Input struct {
	Prompt string
}

type VlmInput struct {
	Prompt string
	Images []llamaserver.ImageData
}

type Sequence struct {
	// batch index
	iBatch int

	// prompt inputs left to evaluate
	inputs []Input

	vlminputs []VlmInput

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	pendingResponses []string

	// channel to send responses over
	responses chan string

	// channel to stop decoding (such as if the remote connection is closed)
	quit chan bool

	// number of tokens to predict
	numPredict int

	// stop sequences
	stop []string

	samplingparameters *SamplingParams

	// number of inputs to keep at the beginning when shifting context window
	numKeep int

	doneReason string

	// tools available for function calling
	tools []api.Tool

	// full chat messages for chat_history-based generation (OpenVINO GenAI)
	messages []api.Message

	// OpenVINO perf metrics: real prompt / generation token counts (set after generate)
	ovPromptTokens int
	ovGenTokens    int
	ovPerfApplied  bool // true after decode path filled metrics from OpenVINO

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptInputs     int
}

func FlushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	seq.pendingResponses = []string{}

	// Check if there are any partial UTF-8 characters remaining.
	// We already check and queue as we are generating but some may
	// still make it here:
	// - Sequence is ending, e.g. generation limit has been hit
	// - Invalid characters in the middle of a string
	// This is a stricter check to ensure we never output invalid Unicode.
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- joined:
		return true
	case <-seq.quit:
		return false
	}
}

func NewSequence(inputs []Input, numPromptInputs int, startTime time.Time,
	samplingParams *SamplingParams, numPredict int, tools []api.Tool) *Sequence {
	return &Sequence{
		inputs:              inputs,
		numPromptInputs:     numPromptInputs,
		startProcessingTime: startTime,
		samplingparameters:  samplingParams,
		numPredict:          numPredict,
		tools:               tools,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
	}
}

func VlmNewSequence(inputs []VlmInput, numPromptInputs int, startTime time.Time,
	samplingParams *SamplingParams, numPredict int, tools []api.Tool) *Sequence {
	return &Sequence{
		vlminputs:           inputs,
		numPromptInputs:     numPromptInputs,
		startProcessingTime: startTime,
		samplingparameters:  samplingParams,
		numPredict:          numPredict,
		tools:               tools,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
	}
}

func (s *Sequence) SetDoneReason(reason string) {
	s.doneReason = reason
}

func (s *Sequence) CloseResponses() {
	close(s.responses)
}

func (s *Sequence) SetStartGenerationTime(t time.Time) {
	s.startGenerationTime = t
}

func (s *Sequence) GetInputs() []Input {
	return s.inputs
}

func (s *Sequence) GetVlmInputs() []VlmInput {
	return s.vlminputs
}

func (i *Input) GetPrompt() string {
	return i.Prompt
}

func (i *VlmInput) GetPrompt() string {
	return i.Prompt
}

func (i *VlmInput) GetImages() []llamaserver.ImageData {
	return i.Images
}

func (s *Sequence) GetSamplingParameters() *SamplingParams {
	return s.samplingparameters
}

func (s *Sequence) AppendPendingResponse(response string) {
	s.pendingResponses = append(s.pendingResponses, response)
}

func (s *Sequence) CloseQuit() {
	close(s.quit)
}

func (s *Sequence) GetResponses() <-chan string {
	return s.responses
}

func (s *Sequence) GetDoneReason() string {
	return s.doneReason
}

func (s *Sequence) GetpendingResponses() []string {
	return s.pendingResponses
}

func (s *Sequence) GetNumPromptInputs() int {
	return s.numPromptInputs
}

func (s *Sequence) GetStartGenerationTime() time.Time {
	return s.startGenerationTime
}

func (s *Sequence) GetStartProcessingTime() time.Time {
	return s.startProcessingTime
}

func (s *Sequence) GetNumDecoded() int {
	return s.numDecoded
}

func (s *Sequence) GetTools() []api.Tool {
	return s.tools
}

func (s *Sequence) GetMessages() []api.Message {
	return s.messages
}

func (s *Sequence) SetMessages(msgs []api.Message) {
	s.messages = msgs
}

// SetOpenVINOTokenCounts stores token counts from ov_genai_perf_metrics after inference.
func (s *Sequence) SetOpenVINOTokenCounts(promptTokens, generationTokens int) {
	s.ovPromptTokens = promptTokens
	s.ovGenTokens = generationTokens
}

// SetOVPerfApplied marks that token counts came from OpenVINO (even if both are zero).
func (s *Sequence) SetOVPerfApplied(applied bool) {
	s.ovPerfApplied = applied
}

// FinalPromptN returns prompt token count for API usage: OpenVINO metrics if available,
// else legacy fallback (number of input segments, often wrong).
func (s *Sequence) FinalPromptN() int {
	if s.ovPerfApplied {
		return s.ovPromptTokens
	}
	return s.numPromptInputs
}

// FinalPredictedN returns generated token count for API usage.
func (s *Sequence) FinalPredictedN() int {
	if s.ovPerfApplied {
		return s.ovGenTokens
	}
	return s.numDecoded
}
