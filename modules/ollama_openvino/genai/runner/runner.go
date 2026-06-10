package runner

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/genai"
	"golang.org/x/sync/semaphore"
)

type NewSequenceParams struct {
	numPredict     int
	stop           []string
	samplingParams *genai.SamplingParams
}

type Server struct {
	// is the server ready to process requests?
	// protects access to model and image
	ready sync.WaitGroup

	// loaded model
	model genai.Model

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

	// indicates that data is ready for processing
	cond *sync.Cond

	// the list of simultaneous sequences being evaluated
	seqs []*(genai.Sequence)

	// seqs can have a maximum of parallel entries, which
	// is enfoced by seqSem
	seqsSem *semaphore.Weighted

	// next sequence for prompt processing to avoid starvation
	nextSeq int

	// device being used
	device string

	// model load error if any
	loadError error
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

func (s *Server) inputs(prompt string, images []ImageData) ([]genai.Input, error) {
	var inputs []genai.Input
	var parts []string

	parts = []string{prompt}

	for _, part := range parts {

		// for _, t := range part {
		// 	inputs = append(inputs, input{prompt: string(t)})
		// }
		inputs = append(inputs, genai.Input{Prompt: string(part)})
	}

	return inputs, nil
}

func (s *Server) NewSequence(prompt string, images []ImageData, params NewSequenceParams) (*(genai.Sequence), error) {
	s.ready.Wait()

	// Check if there was a load error
	if s.loadError != nil {
		return nil, fmt.Errorf("model failed to load: %w", s.loadError)
	}

	startTime := time.Now()

	inputs, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	return genai.NewSequence(inputs, len(inputs), startTime, params.samplingParams, params.numPredict), nil
}

func (s *Server) removeSequence(seqIndex int, reason string) {
	seq := s.seqs[seqIndex]

	genai.FlushPending(seq)
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
		for _, input := range seq.GetInputs() {
			err := genai.GenerateTextWithMetrics(s.model, input.GetPrompt(), seq.GetSamplingParameters(), seq)
			if err != nil {
				log.Printf("Error generating text: %v", err)
				// Set error as done reason so user can see it
				s.removeSequence(i, fmt.Sprintf("error: %v", err))
				return err
			}
			// log.Printf("gen result: ", seq.GetpendingResponses())
		}
		s.removeSequence(i, "")
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

type ImageData struct {
	Data          []byte `json:"data"`
	ID            int    `json:"id"`
	AspectRatioID int    `json:"aspect_ratio_id"`
}

type CompletionRequest struct {
	Prompt      string      `json:"prompt"`
	Images      []ImageData `json:"image_data"`
	Grammar     string      `json:"grammar"`
	CachePrompt bool        `json:"cache_prompt"`

	Options
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

type ErrorResponse struct {
	Error string `json:"error"`
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
	requestDump, err := httputil.DumpRequest(r, true)
	if err != nil {
		log.Println("Error dumping request:", err)
	}
	log.Printf("Request info :\n%s", requestDump)

	var req CompletionRequest
	req.Options = Options(api.DefaultOptions())
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	var samplingParams genai.SamplingParams
	samplingParams.TopK = req.TopK
	samplingParams.TopP = req.TopP
	samplingParams.Temp = req.Temperature
	samplingParams.MaxNewToken = req.MaxNewToken
	samplingParams.StopIds = req.StopId
	samplingParams.StopString = req.Stop
	samplingParams.RepeatLastN = req.RepeatLastN
	samplingParams.RepeatPenalty = req.RepeatPenalty

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict: req.NumPredict,
		// stop:           req.Stop,
		samplingParams: &samplingParams,
	})
	if err != nil {
		log.Printf("Failed to create sequence: %v", err)
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(&ErrorResponse{
			Error: fmt.Sprintf("Failed to create new sequence: %v", err),
		})
		return
	}

	s.mu.Lock()
	found := false

	for i, sq := range s.seqs {
		if sq == nil {
			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}

	s.mu.Unlock()

	if !found {
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

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
				doneReason := seq.GetDoneReason()

				// Check if done reason contains an error
				if strings.HasPrefix(doneReason, "error:") {
					// Send error response
					if err := json.NewEncoder(w).Encode(&CompletionResponse{
						Stop:         true,
						Content:      "",
						StoppedLimit: false,
					}); err != nil {
						http.Error(w, fmt.Sprintf("failed to encode error response: %v", err), http.StatusInternalServerError)
					}
					log.Printf("Generation failed: %s", doneReason)
				} else {
					// Send the final response
					if err := json.NewEncoder(w).Encode(&CompletionResponse{
						Stop:         true,
						StoppedLimit: doneReason == "limit",
						Timings: Timings{
							PromptN:     seq.GetNumPromptInputs(),
							PromptMS:    float64(seq.GetStartGenerationTime().Sub(seq.GetStartProcessingTime()).Milliseconds()),
							PredictedN:  seq.GetNumDecoded(),
							PredictedMS: float64(time.Since(seq.GetStartGenerationTime()).Milliseconds()),
						},
					}); err != nil {
						http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
					}
				}

				return
			}
		}
	}
}

type HealthResponse struct {
	Status   string  `json:"status"`
	Progress float32 `json:"progress"`
	Error    string  `json:"error,omitempty"`
	Device   string  `json:"device,omitempty"`
}

type ServerStatus int

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusLoadingModel
	ServerStatusError
)

func (s ServerStatus) ToString() string {
	switch s {
	case ServerStatusReady:
		return "ok"
	case ServerStatusLoadingModel:
		return "loading model"
	default:
		return "server error"
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := &HealthResponse{
		Status:   s.status.ToString(),
		Progress: s.progress,
		Device:   s.device,
	}

	if s.loadError != nil {
		response.Error = s.loadError.Error()
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
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
	s.device = device

	// Log available devices at startup
	availableDevices := genai.GetAvailableDeviceNames()
	log.Printf("Available devices: %v", availableDevices)
	log.Printf("Requested device: %s", device)

	ov_ir_dir := strings.ReplaceAll(mname, ":", "_")
	tempDir := filepath.Join("/tmp", ov_ir_dir)
	ov_model_path := ""

	isGGUF, err := genai.IsGGUF(mpath)
	if err != nil {
		log.Printf("Error checking if file is GGUF: %v", err)
		s.loadError = fmt.Errorf("error checking file type: %w", err)
		s.status = ServerStatusError
		s.ready.Done()
		return
	}

	if isGGUF {
		log.Printf("The model is a GGUF file.")
		ov_model_path = filepath.Join(tempDir, "tmp.gguf")
		// for GGUF reader
		if _, err := os.Stat(tempDir); os.IsNotExist(err) {
			err := os.MkdirAll(tempDir, 0755)
			if err != nil {
				log.Printf("Error creating directory: %v", err)
				s.loadError = fmt.Errorf("error creating directory: %w", err)
				s.status = ServerStatusError
				s.ready.Done()
				return
			}
			err = genai.CopyFile(mpath, ov_model_path)
			if err != nil {
				log.Printf("Error copying file: %v", err)
				s.loadError = fmt.Errorf("error copying file: %w", err)
				s.status = ServerStatusError
				s.ready.Done()
				return
			}
		}
	}

	isGzip, err := genai.IsGzipByMagicBytes(mpath)
	if err != nil {
		log.Printf("Error checking if file is Gzip: %v", err)
		s.loadError = fmt.Errorf("error checking file type: %w", err)
		s.status = ServerStatusError
		s.ready.Done()
		return
	}

	if isGzip {
		log.Printf("The model is a OpenVINO IR file.")
		// for OpenVINO IR
		_, err = os.Stat(tempDir)
		if os.IsNotExist(err) {
			err = genai.UnpackTarGz(mpath, tempDir)
			if err != nil {
				log.Printf("Error unpacking tar.gz: %v", err)
				s.loadError = fmt.Errorf("error unpacking tar.gz: %w", err)
				s.status = ServerStatusError
				s.ready.Done()
				return
			}
		}

		entries, err := os.ReadDir(tempDir)
		if err != nil {
			log.Printf("Error reading directory: %v", err)
			s.loadError = fmt.Errorf("error reading directory: %w", err)
			s.status = ServerStatusError
			s.ready.Done()
			return
		}

		var subdirs []string
		for _, entry := range entries {
			if entry.IsDir() {
				subdirs = append(subdirs, entry.Name())
			}
		}

		if len(subdirs) == 0 {
			log.Printf("Error: no subdirectories found in unpacked model")
			s.loadError = errors.New("no subdirectories found in unpacked model")
			s.status = ServerStatusError
			s.ready.Done()
			return
		}

		ov_model_path = filepath.Join(tempDir, subdirs[0])
	}

	// Create pipeline with error checking
	model, err := genai.CreatePipeline(ov_model_path, device)
	if err != nil {
		log.Printf("Error creating pipeline: %v", err)
		s.loadError = fmt.Errorf("failed to create pipeline: %w", err)
		s.status = ServerStatusError
		s.ready.Done()
		return
	}

	if model == nil {
		log.Printf("Error: CreatePipeline returned nil model")
		s.loadError = errors.New("pipeline creation returned nil")
		s.status = ServerStatusError
		s.ready.Done()
		return
	}

	s.model = model
	log.Printf("The model has been loaded by GenAI, ov_model_path: %s, device: %s", ov_model_path, device)
	s.status = ServerStatusReady
	s.ready.Done()
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("genairunner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	mname := fs.String("modelname", "", "Name of the model")
	device := fs.String("device", "CPU", "Device to infer")
	parallel := fs.Int("parallel", 1, "Number of sequences to handle simultaneously")
	port := fs.Int("port", 8088, "Port to expose the server on")
	verbose := fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}

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
		seqs:   make([]*genai.Sequence, *parallel),
		status: ServerStatusLoadingModel,
	}

	server.ready.Add(1)
	go server.loadModel(*mpath, *mname, *device)

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

// func main() {
// 	if err := Execute(os.Args[1:]); err != nil {
// 		fmt.Fprintf(os.Stderr, "error: %s\n", err)
// 		os.Exit(1)
// 	}
// }
