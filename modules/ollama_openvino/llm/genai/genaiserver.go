package genaillm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/genai"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	llamaserver "github.com/ollama/ollama/llm/llama"
)

type GenaiServer interface {
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req llamaserver.CompletionRequest, fn func(llamaserver.CompletionResponse)) error
	Close() error
	EstimatedVRAM() uint64 // Total VRAM across all GPUs
	EstimatedTotal() uint64
	EstimatedVRAMByGPU(gpuID string) uint64
}

// GenaillmServer is an instance of the llama.cpp server
type GenaillmServer struct {
	port        int
	cmd         *exec.Cmd
	done        chan error // Channel to signal when the process exits
	status      *llm.StatusWriter
	options     api.Options
	numParallel int
	modelPath   string
	modelName   string
	modelLock   sync.Mutex  // Temporary until we switch fully to Go server
	model       genai.Model // If non-nil, the runner is a new Go server

	estimate    llm.MemoryEstimate
	totalLayers uint64
	// gpuCount     int
	gpus         discover.GpuInfoList // Recorded just before the model loaded, free space will be incorrect
	loadDuration time.Duration        // Record how long it took the model to load
	loadProgress float32

	sem *semaphore.Weighted
}

// LoadModel will load a model from disk. The model must be in the GGML format.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func GenaiLoadModel(model string, maxArraySize int) (*ggml.GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	ggml, _, err := ggml.Decode(f, maxArraySize)
	return ggml, err
}

func ContainsInSlice(items []string, item string) bool {
	for _, eachItem := range items {
		if eachItem == item {
			return true
		}
	}
	return false
}
func SelectDevice(device string, supportedDevices []string) string {
	if device == "" || !ContainsInSlice(supportedDevices, device) {
		if ContainsInSlice(supportedDevices, "GPU") {
			log.Printf("The device specified in the modelfile is not currently supported by GenAI. Now we use GPU")
			return "GPU"
		}
		log.Printf("The device specified in the modelfile is not currently supported by GenAI. Now we use CPU")
		return "CPU"
	}
	return device
}

func addIndexToDuplicates(input []string) []string {
	// output := make([]string, 0, len(input))
	var output []string
	counters := make(map[string]int)    // 用于记录每个值的出现次数
	duplicates := make(map[string]bool) // 用于标记哪些值是重复的

	// 第一次遍历：统计每个值的出现次数，并标记重复值
	for _, item := range input {
		counters[item]++
		if counters[item] > 1 {
			duplicates[item] = true
		}
	}

	// 第二次遍历：为重复值添加索引
	for _, item := range input {
		if duplicates[item] { // 如果是重复值
			output = append(output, fmt.Sprintf("%s:%d", item, counters[item]-1))
			counters[item]-- // 更新计数器
		} else { // 如果不是重复值
			output = append(output, item)
		}
	}
	if ContainsInSlice(input, "GPU") {
		output = append(output, "GPU")
	}

	return output
}

// NewGenaiServer will run a server
func NewGenaiServer(gpus discover.GpuInfoList, model string, modelname string, inferdevice string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (GenaiServer, error) {
	systemInfo := discover.GetSystemInfo()
	systemTotalMemory := systemInfo.System.TotalMemory
	systemFreeMemory := systemInfo.System.FreeMemory
	systemSwapFreeMemory := systemInfo.System.FreeSwap
	slog.Info("system memory", "total", format.HumanBytes2(systemTotalMemory), "free", format.HumanBytes2(systemFreeMemory), "free_swap", format.HumanBytes2(systemSwapFreeMemory))

	genai_device := genai.GetGenaiAvailableDevices()
	var genai_device_list []string
	for i := 0; i < int(len(genai_device)); i++ {
		genai_device_list = append(genai_device_list, genai_device[i]["device_name"])
	}
	genai_device_list = addIndexToDuplicates(genai_device_list)
	inferdevice = SelectDevice(inferdevice, genai_device_list)

	params := []string{
		"--model", model,
		"--modelname", modelname,
		"--device", inferdevice,
		// "--ctx-size", strconv.Itoa(opts.NumCtx),
		// "--batch-size", strconv.Itoa(opts.NumBatch),
	}

	if envconfig.Debug() {
		params = append(params, "--verbose")
	}

	params = append(params, "--parallel", strconv.Itoa(numParallel))

	for {
		port := 0
		if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
			var l *net.TCPListener
			if l, err = net.ListenTCP("tcp", a); err == nil {
				port = l.Addr().(*net.TCPAddr).Port
				l.Close()
			}
		}
		if port == 0 {
			slog.Debug("ResolveTCPAddr failed, using random port")
			port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		}
		finalParams := []string{"genairunner"}
		if envconfig.NewEngine() {
			finalParams = append(finalParams, "--ollama-engine")
		}
		finalParams = append(finalParams, params...)
		finalParams = append(finalParams, "--port", strconv.Itoa(port))

		// log.Printf("---finalParam: %s", finalParams)
		var pathEnv string
		switch runtime.GOOS {
		case "windows":
			pathEnv = "PATH"
		case "darwin":
			pathEnv = "DYLD_LIBRARY_PATH"
		default:
			pathEnv = "LD_LIBRARY_PATH"
		}

		var libraryPaths []string
		if libraryPath, ok := os.LookupEnv(pathEnv); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
		}

		// TODO - once fully switched to the Go runner, load the model here for tokenize/detokenize cgo access
		exe, err := os.Executable()
		if err != nil {
			return nil, fmt.Errorf("unable to lookup executable path: %w", err)
		}

		exe, err = filepath.EvalSymlinks(exe)
		if err != nil {
			return nil, fmt.Errorf("unable to evaluate symlinks for executable path: %w", err)
		}
		s := &GenaillmServer{
			port:      port,
			cmd:       exec.Command(exe, finalParams...),
			status:    llm.NewStatusWriter(os.Stderr),
			options:   opts,
			modelPath: model,
			modelName: modelname,
			// estimate:    estimate,
			numParallel: numParallel,
			sem:         semaphore.NewWeighted(int64(numParallel)),
			// totalLayers: f.KV().BlockCount() + 1,
			// gpus:        gpus,
			done: make(chan error, 1),
		}

		s.cmd.Env = os.Environ()
		s.cmd.Stdout = os.Stdout
		s.cmd.Stderr = s.status
		s.cmd.SysProcAttr = llm.LlamaServerSysProcAttr

		envWorkarounds := [][2]string{}
		for _, gpu := range gpus {
			envWorkarounds = append(envWorkarounds, gpu.EnvWorkarounds...)
		}
		visibleDevicesEnv, visibleDevicesEnvVal := gpus.GetVisibleDevicesEnv()
		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))

		// Update or add the path and visible devices variable with our adjusted version
		pathNeeded := true
		devicesNeeded := visibleDevicesEnv != ""
		for i := range s.cmd.Env {
			cmp := strings.SplitN(s.cmd.Env[i], "=", 2)
			if strings.EqualFold(cmp[0], pathEnv) {
				s.cmd.Env[i] = pathEnv + "=" + pathEnvVal
				pathNeeded = false
			} else if devicesNeeded && strings.EqualFold(cmp[0], visibleDevicesEnv) {
				s.cmd.Env[i] = visibleDevicesEnv + "=" + visibleDevicesEnvVal
				devicesNeeded = false
			} else if len(envWorkarounds) != 0 {
				for _, kv := range envWorkarounds {
					if strings.EqualFold(cmp[0], kv[0]) {
						s.cmd.Env[i] = kv[0] + "=" + kv[1]
					}
				}
			}
		}
		if pathNeeded {
			s.cmd.Env = append(s.cmd.Env, pathEnv+"="+pathEnvVal)
		}
		if devicesNeeded {
			s.cmd.Env = append(s.cmd.Env, visibleDevicesEnv+"="+visibleDevicesEnvVal)
		}

		slog.Info("starting llama server", "cmd", s.cmd.String())

		if err = s.cmd.Start(); err != nil {
			var msg string
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			err := fmt.Errorf("error starting runner: %v %s", err, msg)
			return nil, err
		}

		// reap subprocess when it exits
		go func() {
			err := s.cmd.Wait()
			// Favor a more detailed message over the process exit status
			if err != nil && s.status != nil && s.status.LastErrMsg != "" {
				slog.Error("llama runner terminated", "error", err)
				if strings.Contains(s.status.LastErrMsg, "unknown model") {
					s.status.LastErrMsg = "this model is not supported by your version of Ollama. You may need to upgrade"
				}
				s.done <- errors.New(s.status.LastErrMsg)
			} else {
				s.done <- err
			}
		}()

		return s, nil
	}
}

func (s *GenaillmServer) getServerStatus(ctx context.Context) (llamaserver.ServerStatus, error) {
	// Fail fast if its exited
	if s.cmd.ProcessState != nil {
		msg := ""
		if s.status != nil && s.status.LastErrMsg != "" {
			msg = s.status.LastErrMsg
		}
		if s.cmd.ProcessState.ExitCode() == -1 {
			// Most likely a signal killed it, log some more details to try to help troubleshoot
			slog.Warn("llama runner process no longer running", "sys", s.cmd.ProcessState.Sys(), "string", s.cmd.ProcessState.String())
		}
		return llamaserver.ServerStatusError, fmt.Errorf("llama runner process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/health", s.port), nil)
	if err != nil {
		return llamaserver.ServerStatusError, fmt.Errorf("error creating GET request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			return llamaserver.ServerStatusNotResponding, errors.New("server not responding")
		}
		return llamaserver.ServerStatusError, fmt.Errorf("health resp: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return llamaserver.ServerStatusError, fmt.Errorf("read health request: %w", err)
	}

	var status llamaserver.ServerStatusResp
	if err := json.Unmarshal(body, &status); err != nil {
		return llamaserver.ServerStatusError, fmt.Errorf("health unmarshal encode response: %w", err)
	}

	switch status.Status {
	case "ok":
		return llamaserver.ServerStatusReady, nil
	case "no slot available":
		return llamaserver.ServerStatusNoSlotsAvailable, nil
	case "loading model":
		s.loadProgress = status.Progress
		return llamaserver.ServerStatusLoadingModel, nil
	default:
		return llamaserver.ServerStatusError, fmt.Errorf("server error: %+v", status)
	}
}

// getServerStatusRetry will retry if ServerStatusNoSlotsAvailable is received
func (s *GenaillmServer) getServerStatusRetry(ctx context.Context) (llamaserver.ServerStatus, error) {
	var retries int
	for {
		status, err := s.getServerStatus(ctx)
		if err != nil {
			return status, err
		}

		if status == llamaserver.ServerStatusNoSlotsAvailable {
			if retries >= 10 {
				return status, fmt.Errorf("no slots available after %d retries", retries)
			}

			time.Sleep(5 * time.Millisecond)
			retries++
			continue
		}

		return status, nil
	}
}

func (s *GenaillmServer) Ping(ctx context.Context) error {
	_, err := s.getServerStatus(ctx)
	if err != nil {
		slog.Debug("server unhealthy", "error", err)
		return err
	}
	return nil
}

func (s *GenaillmServer) WaitUntilRunning(ctx context.Context) error {
	start := time.Now()
	stallDuration := envconfig.LoadTimeout()    // If no progress happens
	stallTimer := time.Now().Add(stallDuration) // give up if we stall

	slog.Info("waiting for llama runner to start responding")
	var lastStatus llamaserver.ServerStatus = -1
	fullyLoaded := false

	for {
		select {
		case <-ctx.Done():
			slog.Warn("client connection closed before server finished loading, aborting load")
			return fmt.Errorf("timed out waiting for llama runner to start: %w", ctx.Err())
		case err := <-s.done:
			return fmt.Errorf("llama runner process has terminated: %w", err)
		default:
		}
		if time.Now().After(stallTimer) {
			// timeout
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("timed out waiting for llama runner to start - progress %0.2f - %s", s.loadProgress, msg)
		}
		if s.cmd.ProcessState != nil {
			msg := ""
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			}
			return fmt.Errorf("llama runner process no longer running: %d %s", s.cmd.ProcessState.ExitCode(), msg)
		}
		ctx, cancel := context.WithTimeout(ctx, 200*time.Millisecond)
		defer cancel()
		priorProgress := s.loadProgress
		status, _ := s.getServerStatus(ctx)
		if lastStatus != status && status != llamaserver.ServerStatusReady {
			// Only log on status changes
			slog.Info("waiting for server to become available", "status", status.ToString())
		}
		switch status {
		case llamaserver.ServerStatusReady:
			s.loadDuration = time.Since(start)
			slog.Info(fmt.Sprintf("llama runner started in %0.2f seconds", s.loadDuration.Seconds()))
			return nil
		default:
			lastStatus = status
			// Reset the timer as long as we're making forward progress on the load
			if priorProgress != s.loadProgress {
				slog.Debug(fmt.Sprintf("model load progress %0.2f", s.loadProgress))
				stallTimer = time.Now().Add(stallDuration)
			} else if !fullyLoaded && int(s.loadProgress*100.0) >= 100 {
				slog.Debug("model load completed, waiting for server to become available", "status", status.ToString())
				stallTimer = time.Now().Add(stallDuration)
				fullyLoaded = true
			}
			time.Sleep(time.Millisecond * 250)
			continue
		}
	}
}

var grammarJSON = `
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= ([ \t\n] ws)?
`

const maxBufferSize = 512 * format.KiloByte

type completion struct {
	Content      string `json:"content"`
	Model        string `json:"model"`
	Prompt       string `json:"prompt"`
	Stop         bool   `json:"stop"`
	StoppedLimit bool   `json:"stopped_limit"`

	Timings struct {
		PredictedN  int     `json:"predicted_n"`
		PredictedMS float64 `json:"predicted_ms"`
		PromptN     int     `json:"prompt_n"`
		PromptMS    float64 `json:"prompt_ms"`
	}
}

func (s *GenaillmServer) Completion(ctx context.Context, req llamaserver.CompletionRequest, fn func(llamaserver.CompletionResponse)) error {
	request := map[string]any{
		"prompt":            req.Prompt,
		"stream":            true,
		"max_new_token":     req.Options.MaxNewToken,
		"stop_id":           req.Options.StopId,
		"n_predict":         req.Options.NumPredict,
		"n_keep":            req.Options.NumKeep,
		"main_gpu":          req.Options.MainGPU,
		"temperature":       req.Options.Temperature,
		"top_k":             req.Options.TopK,
		"top_p":             req.Options.TopP,
		"min_p":             req.Options.MinP,
		"typical_p":         req.Options.TypicalP,
		"repeat_last_n":     req.Options.RepeatLastN,
		"repeat_penalty":    req.Options.RepeatPenalty,
		"presence_penalty":  req.Options.PresencePenalty,
		"frequency_penalty": req.Options.FrequencyPenalty,
		"mirostat":          req.Options.Mirostat,
		"mirostat_tau":      req.Options.MirostatTau,
		"mirostat_eta":      req.Options.MirostatEta,
		"seed":              req.Options.Seed,
		"stop":              req.Options.Stop,
		"image_data":        req.Images,
		"cache_prompt":      true,
	}

	if len(req.Format) > 0 {
		switch string(req.Format) {
		case `null`, `""`:
			// Field was set, but "missing" a value. We accept
			// these as "not set".
			break
		case `"json"`:
			request["grammar"] = grammarJSON
		default:
			if req.Format[0] != '{' {
				return fmt.Errorf("invalid format: %q; expected \"json\" or a valid JSON Schema object", req.Format)
			}

			// User provided a JSON schema
			g := llama.SchemaToGrammar(req.Format)
			if g == nil {
				return fmt.Errorf("invalid JSON schema in format")
			}
			request["grammar"] = string(g)
		}
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			slog.Error("Failed to acquire semaphore", "error", err)
		}
		return err
	}
	defer s.sem.Release(1)

	// put an upper limit on num_predict to avoid the model running on forever
	if req.Options.NumPredict < 0 || req.Options.NumPredict > 10*s.options.NumCtx {
		req.Options.NumPredict = 10 * s.options.NumCtx
	}

	// Make sure the server is ready
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return err
	} else if status != llamaserver.ServerStatusReady {
		return fmt.Errorf("unexpected server status: %s", status.ToString())
	}

	// Handling JSON marshaling with special characters unescaped.
	buffer := &bytes.Buffer{}
	enc := json.NewEncoder(buffer)
	enc.SetEscapeHTML(false)

	if err := enc.Encode(request); err != nil {
		return fmt.Errorf("failed to marshal data: %v", err)
	}

	endpoint := fmt.Sprintf("http://127.0.0.1:%d/completion", s.port)
	serverReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buffer)
	if err != nil {
		return fmt.Errorf("error creating POST request: %v", err)
	}
	serverReq.Header.Set("Content-Type", "application/json")

	res, err := http.DefaultClient.Do(serverReq)
	if err != nil {
		return fmt.Errorf("POST predict: %v", err)
	}
	defer res.Body.Close()

	if res.StatusCode >= 400 {
		bodyBytes, err := io.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("failed reading llm error response: %w", err)
		}
		log.Printf("llm predict error: %s", bodyBytes)
		return fmt.Errorf("%s", bodyBytes)
	}

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, maxBufferSize)
	scanner.Buffer(buf, maxBufferSize)

	// keep track of the last token generated, this is used to abort if the model starts looping
	var lastToken string
	var tokenRepeat int

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			// This handles the request cancellation
			return ctx.Err()
		default:
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			// slog.Debug("got line", "line", string(line))
			evt, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok {
				evt = line
			}

			var c completion
			if err := json.Unmarshal(evt, &c); err != nil {
				return fmt.Errorf("error unmarshalling llm prediction response: %v", err)
			}
			switch {
			case strings.TrimSpace(c.Content) == lastToken:
				tokenRepeat++
			default:
				lastToken = strings.TrimSpace(c.Content)
				tokenRepeat = 0
			}

			// 30 picked as an arbitrary max token repeat limit, modify as needed
			if tokenRepeat > 30 {
				slog.Debug("prediction aborted, token repeat limit reached")
				return ctx.Err()
			}

			if c.Content != "" {
				fn(llamaserver.CompletionResponse{
					Content: c.Content,
				})
			}

			if c.Stop {
				doneReason := "stop"
				if c.StoppedLimit {
					doneReason = "length"
				}

				fn(llamaserver.CompletionResponse{
					Done:               true,
					DoneReason:         doneReason,
					PromptEvalCount:    c.Timings.PromptN,
					PromptEvalDuration: parseDurationMs(c.Timings.PromptMS),
					EvalCount:          c.Timings.PredictedN,
					EvalDuration:       parseDurationMs(c.Timings.PredictedMS),
				})
				return nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if strings.Contains(err.Error(), "unexpected EOF") || strings.Contains(err.Error(), "forcibly closed") {
			s.Close()
			var msg string
			if s.status != nil && s.status.LastErrMsg != "" {
				msg = s.status.LastErrMsg
			} else {
				msg = err.Error()
			}
			return fmt.Errorf("an error was encountered while running the model: %s", msg)
		}

		return fmt.Errorf("error reading llm response: %v", err)
	}

	return nil
}

func (s *GenaillmServer) Clear(mname string) error {
	var err error
	ov_ir_dir := strings.ReplaceAll(mname, ":", "_")
	modelDir := filepath.Join("/tmp", ov_ir_dir)
	err = os.RemoveAll(modelDir)
	if err != nil {
		fmt.Println("Error removing directory:", err)
		return nil
	}
	return nil
}

func (s *GenaillmServer) Close() error {
	s.modelLock.Lock()
	if s.modelName != "" {
		genai.FreeModel(s.model)
	}
	s.modelLock.Unlock()

	if s.cmd != nil {
		slog.Debug("stopping llama server")
		if s.cmd.ProcessState != nil && s.cmd.ProcessState.Exited() {
			s.Clear(s.modelName)
			s.modelName = ""
			return nil
		} else {
			if s.cmd.ProcessState == nil {
				if err := s.cmd.Process.Kill(); err != nil {
					s.cmd.Wait()
				}
				// if ProcessState is already populated, Wait already completed, no need to wait again
				if s.cmd.ProcessState == nil {
					slog.Debug("waiting for llama server to exit")
					<-s.done
				}

				slog.Debug("llama server stopped")
			}
		}
	}
	s.Clear(s.modelName)
	s.modelName = ""

	return nil
}

func (s *GenaillmServer) EstimatedVRAM() uint64 {
	return s.estimate.VRAMSize
}

func (s *GenaillmServer) EstimatedTotal() uint64 {
	return s.estimate.TotalSize
}

func (s *GenaillmServer) EstimatedVRAMByGPU(gpuID string) uint64 {
	for i, gpu := range s.gpus {
		if gpu.ID == gpuID {
			if i < len(s.estimate.GPUSizes) {
				return s.estimate.GPUSizes[i]
			}
		}
	}
	return 0
}

func parseDurationMs(ms float64) time.Duration {
	dur, err := time.ParseDuration(fmt.Sprintf("%fms", ms))
	if err != nil {
		panic(err)
	}

	return dur
}
