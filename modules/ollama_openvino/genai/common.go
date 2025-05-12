package genai

import (
	"strings"
	"time"
	"unicode/utf8"
)

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type Input struct {
	Prompt string
}

type Sequence struct {
	// batch index
	iBatch int

	// prompt inputs left to evaluate
	inputs []Input

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

// NewSequence 创建一个新的 Sequence 实例
func NewSequence(inputs []Input, numPromptInputs int, startTime time.Time,
	samplingParams *SamplingParams, numPredict int) *Sequence {
	return &Sequence{
		inputs:              inputs,
		numPromptInputs:     numPromptInputs,
		startProcessingTime: startTime,
		samplingparameters:  samplingParams,
		numPredict:          numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
	}
}

// SetDoneReason 设置 Sequence 的 doneReason 字段
func (s *Sequence) SetDoneReason(reason string) {
	s.doneReason = reason
}

// CloseResponses 关闭 Sequence 的 responses 通道
func (s *Sequence) CloseResponses() {
	close(s.responses)
}

// SetStartGenerationTime 设置 Sequence 的 startGenerationTime 字段
func (s *Sequence) SetStartGenerationTime(t time.Time) {
	s.startGenerationTime = t
}

// GetInputs 返回 Sequence 的 inputs 字段
func (s *Sequence) GetInputs() []Input {
	return s.inputs
}

// GetPrompt 返回 input 的 prompt 字段
func (i *Input) GetPrompt() string {
	return i.Prompt
}

// GetSamplingParameters 返回 Sequence 的 samplingparameters 字段
func (s *Sequence) GetSamplingParameters() *SamplingParams {
	return s.samplingparameters
}

// AppendPendingResponse 向 Sequence 的 pendingResponses 追加内容
func (s *Sequence) AppendPendingResponse(response string) {
	s.pendingResponses = append(s.pendingResponses, response)
}

// CloseQuit 关闭 Sequence 的 quit 通道
func (s *Sequence) CloseQuit() {
	close(s.quit)
}

// GetResponses 返回 Sequence 的 responses 通道
func (s *Sequence) GetResponses() <-chan string {
	return s.responses
}

// GetDoneReason 返回 Sequence 的 doneReason 字段
func (s *Sequence) GetDoneReason() string {
	return s.doneReason
}

// GetDoneReason 返回 Sequence 的 doneReason 字段
func (s *Sequence) GetpendingResponses() []string {
	return s.pendingResponses
}

// GetNumPromptInputs 返回 Sequence 的 numPromptInputs 字段
func (s *Sequence) GetNumPromptInputs() int {
	return s.numPromptInputs
}

// GetStartGenerationTime 返回 Sequence 的 startGenerationTime 字段
func (s *Sequence) GetStartGenerationTime() time.Time {
	return s.startGenerationTime
}

// GetStartProcessingTime 返回 Sequence 的 startProcessingTime 字段
func (s *Sequence) GetStartProcessingTime() time.Time {
	return s.startProcessingTime
}

// GetNumDecoded 返回 Sequence 的 numDecoded 字段
func (s *Sequence) GetNumDecoded() int {
	return s.numDecoded
}
