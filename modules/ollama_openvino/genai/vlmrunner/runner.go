package vlmrunner

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/genai/common"
	llamaserver "github.com/ollama/ollama/llm/llama"
	"golang.org/x/sync/semaphore"
)

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	samplingParams *common.SamplingParams
	tools          []api.Tool
	messages       []api.Message
}

type Server struct {
	// is the server ready to process requests?
	// protects access to model and image
	ready sync.WaitGroup

	// loaded model
	model common.Model

	vlmmodel common.VlmModel

	// status for external health reporting - loading, ready to serve, etc.
	status ServerStatus

	// current progress on loading the model
	progress float32

	// number of simultaneous requests to handle
	parallel int

	// maximum number of elements in a batch (per sequence)
	// TODO (jmorganca): make this n_batch
	batchSize int

	// protects access to everything below this line
	// this is context state needed for decoding
	mu sync.Mutex

	// protects load operations
	loadMu sync.Mutex

	// indicates that data is ready for processing
	cond *sync.Cond

	// the list of simultaneous sequences being evaluated
	seqs []*(common.Sequence)

	// seqs can have a maximum of parallel entries, which
	// is enfoced by seqSem
	seqsSem *semaphore.Weighted

	// next sequence for prompt processing to avoid starvation
	nextSeq int

	// number of requests currently queued in completion() waiting for a free
	// sequence slot (guarded by mu) — surfaced in logs for visibility.
	queued int

	// model metadata for load operation
	modelPath   string
	modelName   string
	inferDevice string
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func (s *Server) inputs(prompt string, images []llamaserver.ImageData) ([]common.VlmInput, error) {
	var inputs []common.VlmInput
	var parts []string

	parts = []string{prompt}

	for _, part := range parts {

		// for _, t := range part {
		// 	inputs = append(inputs, input{prompt: string(t)})
		// }
		inputs = append(inputs, common.VlmInput{Prompt: string(part), Images: images})
	}

	return inputs, nil
}

func (s *Server) NewSequence(prompt string, images []llamaserver.ImageData, params NewSequenceParams) (*(common.Sequence), error) {
	s.ready.Wait()

	startTime := time.Now()

	inputs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	seq := common.VlmNewSequence(inputs, len(inputs), startTime, params.samplingParams, params.numPredict, params.tools)
	if len(params.messages) > 0 {
		seq.SetMessages(params.messages)
	}
	return seq, nil
}

func (s *Server) removeSequence(seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	common.FlushPending(seq)
	seq.SetDoneReason(reason)
	seq.CloseResponses()
	s.seqs[seqIndex] = nil
}

// TODO (jmorganca): processBatch should be simplified, removing:
// * sampling
// * stop token checking
// * metrics
// these should instead be handled by the handlers
// it should only be responsible for accepting tokens or embeddings and
// processing batches as fast as possible
func (s *Server) processBatch() error {
	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	seqIdx := s.nextSeq - 1
	for i, seq := range s.seqs {
		if seq == nil {
			continue
		}

		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]

		if seq == nil {
			continue
		}

		seq.SetStartGenerationTime(time.Now())
		if tools := seq.GetTools(); len(tools) > 0 {
			log.Printf("vlm tools available: %d", len(tools))
			// for _, t := range tools {
			// 	log.Printf("  tool: %s", t.Function.Name)
			// }
		}
		if msgs := seq.GetMessages(); len(msgs) > 0 {
			lastImages := lastUserImagesFromMessages(msgs)
			if len(lastImages) == 0 {
				for _, in := range seq.GetVlmInputs() {
					if len(in.GetImages()) > 0 {
						lastImages = in.GetImages()
						break
					}
				}
			}
			log.Printf("vlm generating with chat history, %d messages, %d images (last user / fallback)", len(msgs), len(lastImages))
			common.VlmGenerateWithChatHistory(s.vlmmodel, msgs, lastImages, seq.GetTools(), seq.GetSamplingParameters(), seq)
		} else {
			for _, input := range seq.GetVlmInputs() {
				common.VlmGenerateTextWithMetrics(s.vlmmodel, input.GetPrompt(), input.GetImages(), seq.GetSamplingParameters(), seq)
			}
		}
		s.removeSequence(i, "")
		// A slot just freed — wake any requests queued in completion().
		s.cond.Broadcast()
	}
	return nil
}

// TODO (jmorganca): use structs from the api package to avoid duplication
// this way the api acts as a proxy instead of using a different api for the
// runner
type Options struct {
	api.Runner

	NumKeep          int      `json:"n_keep"`
	Seed             int      `json:"seed"`
	NumPredict       int      `json:"n_predict"`
	TopK             int      `json:"top_k"`
	TopP             float32  `json:"top_p"`
	StopId           []string `json:"stop_id,omitempty"`
	MaxNewToken      int      `json:"max_new_token,omitempty"`
	MinP             float32  `json:"min_p"`
	TypicalP         float32  `json:"typical_p"`
	RepeatLastN      int      `json:"repeat_last_n"`
	Temperature      float32  `json:"temperature"`
	RepeatPenalty    float32  `json:"repeat_penalty"`
	PresencePenalty  float32  `json:"presence_penalty"`
	FrequencyPenalty float32  `json:"frequency_penalty"`
	Mirostat         int      `json:"mirostat"`
	MirostatTau      float32  `json:"mirostat_tau"`
	MirostatEta      float32  `json:"mirostat_eta"`
	Stop             []string `json:"stop"`
}

// type ImageData struct {
// 	Data          []byte `json:"data"`
// 	ID            int    `json:"id"`
// 	AspectRatioID int    `json:"aspect_ratio_id"`
// }

type CompletionRequest struct {
	Prompt      string                  `json:"prompt"`
	Images      []llamaserver.ImageData `json:"image_data"`
	Grammar     string                  `json:"grammar"`
	CachePrompt bool                    `json:"cache_prompt"`
	Tools       []api.Tool              `json:"tools,omitempty"`
	// Messages use OpenVINO chat_history + generate_with_history. RGB tensors attach to the
	// last user turn (see lastUserImagesFromMessages); else top-level image_data is used.
	Messages []api.Message `json:"messages,omitempty"`

	Options *api.Options
}

type Timings struct {
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
}

type CompletionResponse struct {
	Content string `json:"content"`
	Stop    bool   `json:"stop"`

	Model        string  `json:"model,omitempty"`
	Prompt       string  `json:"prompt,omitempty"`
	StoppedLimit bool    `json:"stopped_limit,omitempty"`
	PredictedN   int     `json:"predicted_n,omitempty"`
	PredictedMS  float64 `json:"predicted_ms,omitempty"`
	PromptN      int     `json:"prompt_n,omitempty"`
	PromptMS     float64 `json:"prompt_ms,omitempty"`

	Timings Timings `json:"timings"`
}

// func (s *Server) run(ctx context.Context) {
func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	for {
		select {
		case <-ctx.Done():
			return
		default:
			s.processBatch()
		}
	}
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	var req CompletionRequest
	opts := api.DefaultOptions()
	req.Options = &opts

	if err := json.Unmarshal(body, &req); err != nil {
		log.Printf("decode request failed: %T: %v", err, err)
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	var rawMap map[string]any
	if err := json.Unmarshal(body, &rawMap); err == nil {
		if err := req.Options.FromMap(rawMap); err != nil {
			log.Printf("warning: failed to parse options from request: %v", err)
		}
	}

	if len(req.Messages) > 0 {
		log.Printf("vlm completion: %d messages, prompt_len=%d image_data=%d",
			len(req.Messages), len(req.Prompt), len(req.Images))
	}

	// if b, err := json.MarshalIndent(req, "", "  "); err == nil {
	// 	log.Printf("req(json) = %s", b)
	// }

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	var samplingParams common.SamplingParams
	samplingParams.TopK = req.Options.TopK
	samplingParams.TopP = req.Options.TopP
	samplingParams.Temp = req.Options.Temperature
	samplingParams.MaxNewToken = req.Options.MaxNewToken
	samplingParams.StopIds = req.Options.StopId
	samplingParams.StopString = req.Options.Stop
	samplingParams.RepeatLastN = req.Options.RepeatLastN
	samplingParams.RepeatPenalty = req.Options.RepeatPenalty
	samplingParams.EnableThinking = req.Options.EnableThinking

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:     req.Options.NumPredict,
		samplingParams: &samplingParams,
		tools:          req.Tools,
		messages:       req.Messages,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Claim a sequence slot, WAITING for one to free up rather than rejecting.
	// The number of slots (len(s.seqs) == --parallel) bounds how many requests
	// are in flight at once; the rest queue here. A request whose client
	// disconnects (ctx cancelled) stops waiting instead of blocking forever —
	// a watcher goroutine broadcasts the cond on cancellation to wake us.
	reqCtx := r.Context()
	watchDone := make(chan struct{})
	go func() {
		select {
		case <-reqCtx.Done():
			s.mu.Lock()
			s.cond.Broadcast()
			s.mu.Unlock()
		case <-watchDone:
		}
	}()
	defer close(watchDone)

	s.mu.Lock()
	claimed := false
	counted := false // whether we've added ourselves to the queued tally
	for {
		if reqCtx.Err() != nil {
			if counted {
				s.queued--
			}
			log.Printf("vlm queue: request cancelled while waiting (%d still queued)", s.queued)
			s.mu.Unlock()
			return // client gave up while queued
		}
		for i, sq := range s.seqs {
			if sq == nil {
				s.seqs[i] = seq
				s.cond.Signal() // wake the run loop to process it
				claimed = true
				break
			}
		}
		if claimed {
			if counted {
				s.queued--
			}
			break
		}
		// No free slot: join the queue (count once) and wait.
		if !counted {
			s.queued++
			counted = true
			log.Printf("vlm queue: no free slot, request waiting (%d now queued, %d slots)", s.queued, len(s.seqs))
		}
		s.cond.Wait() // wait for a slot to free (or for ctx cancel)
	}
	s.mu.Unlock()

	for {
		select {
		case <-r.Context().Done():
			seq.CloseQuit()
			return
		case content, ok := <-seq.GetResponses():
			if ok {
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Content: content,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					seq.CloseQuit()
					return
				}

				flusher.Flush()
			} else {
				// Send the final response
				if err := json.NewEncoder(w).Encode(&CompletionResponse{
					Stop:         true,
					StoppedLimit: seq.GetDoneReason() == "limit",
					Timings: Timings{
						PromptN:     seq.FinalPromptN(),
						PromptMS:    float64(seq.GetStartGenerationTime().Sub(seq.GetStartProcessingTime()).Milliseconds()),
						PredictedN:  seq.FinalPredictedN(),
						PredictedMS: float64(time.Since(seq.GetStartGenerationTime()).Milliseconds()),
					},
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
	}
}

type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
}

type ServerStatus int

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusLoadingModel
	ServerStatusLaunched
	ServerStatusError
)

func (s ServerStatus) ToString() string {
	switch s {
	case ServerStatusReady:
		return "ok"
	case ServerStatusLoadingModel:
		return "loading model"
	case ServerStatusLaunched:
		return "launched"
	default:
		return "server error"
	}
}

// LoadRequest represents the request to load a model
type LoadRequest struct {
	ModelPath   string `json:"model_path"`
	ShortName   string `json:"short_name"`
	ModelType   string `json:"model_type"`
	InferDevice string `json:"infer_device"`
}

// LoadResponse represents the response after loading a model
type LoadResponse struct {
	Success bool `json:"success"`
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var llamaStatus llamaserver.ServerStatus
	switch s.status {
	case ServerStatusReady:
		llamaStatus = llamaserver.ServerStatusReady
	case ServerStatusLoadingModel:
		llamaStatus = llamaserver.ServerStatusLoadingModel
	case ServerStatusLaunched:
		llamaStatus = llamaserver.ServerStatusLaunched
	case ServerStatusError:
		llamaStatus = llamaserver.ServerStatusError
	default:
		llamaStatus = llamaserver.ServerStatusError
	}

	if err := json.NewEncoder(w).Encode(&llamaserver.ServerStatusResponse{
		Status:   llamaStatus,
		Progress: s.progress,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	var req LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	slog.Info("load", "request", req)

	s.modelPath = req.ModelPath
	s.modelName = req.ShortName
	s.inferDevice = req.InferDevice

	if s.status == ServerStatusLoadingModel || s.status == ServerStatusReady {
		resp := LoadResponse{Success: true}
		if err := json.NewEncoder(w).Encode(&resp); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.status = ServerStatusLoadingModel
	go s.loadModel(req.ModelPath, req.ShortName, req.InferDevice)

	resp := LoadResponse{Success: true}
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

type multiLPath []string

func (m *multiLPath) Set(value string) error {
	*m = append(*m, value)
	return nil
}

func (m *multiLPath) String() string {
	return strings.Join(*m, ", ")
}

func (s *Server) loadModel(mpath string, mname string, device string) {
	var err error
	ov_ir_dir := strings.ReplaceAll(mname, ":", "_")
	log.Printf("ov_ir_dir: %s\n", ov_ir_dir)
	tempDir := filepath.Join("/tmp", ov_ir_dir)
	ov_model_path := ""

	log.Printf("tempDir: %s\n", tempDir)

	isGzip, err := common.IsGzipByMagicBytes(mpath)
	if err != nil {
		fmt.Printf("Error checking file: %v\n", err)
	}
	if isGzip {
		log.Printf("The model is a OpenVINO IR file.")
		// for OpenVINO IR
		_, err = os.Stat(tempDir)
		if os.IsNotExist(err) {
			err = common.UnpackTarGz(mpath, tempDir)
			if err != nil {
				panic(err)
			}
		}

		entries, _ := os.ReadDir(tempDir)
		var subdirs []string
		for _, entry := range entries {
			if entry.IsDir() {
				subdirs = append(subdirs, entry.Name())
			}
		}

		ov_model_path = filepath.Join(tempDir, subdirs[0])
	}

	s.vlmmodel = common.CreateVlmPipeline(ov_model_path, device)
	log.Printf("The model had been load by GenAI, ov_model_path: %s , %s", ov_model_path, device)
	s.status = ServerStatusReady
	s.ready.Done()
}

func Execute(args []string) error {
	log.Printf("VLM Execute called with args: %+v", args)

	fs := flag.NewFlagSet("genairunner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	mname := fs.String("modelname", "", "Name of the model")
	device := fs.String("device", "GPU", "Device to infer")
	parallel := fs.Int("parallel", 1, "Number of sequences to handle simultaneously")
	port := fs.Int("port", 8088, "Port to expose the server on")
	verbose := fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}

	log.Printf("Parsed flags - model: %s, modelname: %s, device: %s", *mpath, *mname, *device)

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})
	slog.SetDefault(slog.New(handler))
	slog.Info("starting go genairunner")

	server := &Server{
		seqs:   make([]*common.Sequence, *parallel),
		status: ServerStatusLaunched,
	}

	server.ready.Add(1)

	server.cond = sync.NewCond(&server.mu)

	ctx, cancel := context.WithCancel(context.Background())
	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		cancel()
		return err
	}
	defer listener.Close()

	mux := http.NewServeMux()
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("/completion", server.completion)
	mux.HandleFunc("/health", server.health)

	httpServer := http.Server{
		Handler: mux,
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
		return err
	}

	fmt.Println("after server")
	cancel()
	return nil
}

// lastUserImagesFromMessages returns image data for the chronologically last user message only
// (OpenVINO VLM binds rgbs to the latest user turn). If that turn has no images, returns nil.
func lastUserImagesFromMessages(msgs []api.Message) []llamaserver.ImageData {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		if len(msgs[i].Images) == 0 {
			return nil
		}
		images := make([]llamaserver.ImageData, 0, len(msgs[i].Images))
		for j, raw := range msgs[i].Images {
			images = append(images, llamaserver.ImageData{ID: j, Data: raw})
		}
		return images
	}
	return nil
}

// func main() {
// 	if err := Execute(os.Args[1:]); err != nil {
// 		fmt.Fprintf(os.Stderr, "error: %s\n", err)
// 		os.Exit(1)
// 	}
// }
