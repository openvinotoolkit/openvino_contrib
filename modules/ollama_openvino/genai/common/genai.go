package common

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17

#cgo LDFLAGS: -lopenvino_genai
#cgo LDFLAGS: -lopenvino
#cgo LDFLAGS: -lopenvino_c
#cgo LDFLAGS: -lopenvino_genai_c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openvino/genai/c/llm_pipeline.h"
#include "openvino/genai/c/vlm_pipeline.h"
#include "openvino/genai/c/visibility.h"
#include "openvino/genai/c/chat_history.h"
#include "openvino/genai/c/json_container.h"

#include "openvino/c/openvino.h"
#include "openvino/c/ov_common.h"
#include <stdbool.h>

typedef int (*callback_function)(const char*, void*);

extern int goCallbackBridge(char* input, void* ptr);

// Compatibility macros using numeric values (same in both versions)
#define OV_GENAI_STREAMING_STATUS_RUNNING_COMPAT 0 // Continue to run inference
#define OV_GENAI_STREAMING_STATUS_STOP_COMPAT 1 // Stop generation, keep history
#define OV_GENAI_STREAMING_STATUS_CANCEL_COMPAT 2 // Stop and drop last prompt

static ov_status_e ov_genai_llm_pipeline_create_npu_output_2048(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, "NPU", 4, pipe, "MAX_PROMPT_LEN", "2048", "MIN_RESPONSE_LEN", "256");
}

static ov_status_e ov_genai_llm_pipeline_create_cgo(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, device, 0, pipe);
}

static ov_status_e ov_genai_vlm_pipeline_create_cgo(const char* models_path,
																  const char* device,
                                                                  ov_genai_vlm_pipeline** pipe) {
	return 	ov_genai_vlm_pipeline_create(models_path, device, 0, pipe);
}

*/
import "C"

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // 支持 GIF
	_ "image/jpeg" // 支持 JPEG
	_ "image/png"  // 支持 PNG
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/cgo"
	"strconv"
	"unsafe"

	"github.com/ollama/ollama/api"
	llamaserver "github.com/ollama/ollama/llm/llama"
)

// NOTE on thread-safety: the OpenVINO GenAI LLM/VLM pipelines are NOT
// thread-safe — each holds its own inference and KV-cache state and allows only
// one in-flight generate. These Generate* functions therefore must be called
// serially. That is guaranteed by the runners: every generate happens inside
// processBatch, which runs in a single run() goroutine and holds Server.mu for
// the whole call. Do NOT call generate from any other goroutine/path without
// adding serialization here.

type SamplingParams struct {
	TopK           int
	TopP           float32
	Temp           float32
	MaxNewToken    int
	RepeatPenalty  float32
	StopString     []string
	StopIds        []string
	RepeatLastN    int
	EnableThinking bool
}

// type Model struct {
// 	pipe *C.LLMPipelineHandle
// }

type Model *C.ov_genai_llm_pipeline
type VlmModel *C.ov_genai_vlm_pipeline

func IsGGUF(filePath string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read the first 4 bytes (magic number for GGUF)
	reader := bufio.NewReader(file)
	magicBytes := make([]byte, 4)
	_, err = reader.Read(magicBytes)
	if err != nil {
		return false, fmt.Errorf("failed to read magic number: %v", err)
	}

	// Compare the magic number (GGUF in ASCII)
	expectedMagic := []byte{0x47, 0x47, 0x55, 0x46} // "GGUF" in hex
	for i := 0; i < 4; i++ {
		if magicBytes[i] != expectedMagic[i] {
			return false, nil
		}
	}

	return true, nil
}

func IsGzipByMagicBytes(filepath string) (bool, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return false, err
	}
	defer file.Close()

	magicBytes := make([]byte, 2)
	_, err = file.Read(magicBytes)
	if err != nil {
		return false, err
	}

	return bytes.Equal(magicBytes, []byte{0x1F, 0x8B}), nil
}

func CopyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return err
	}

	err = dstFile.Sync()
	if err != nil {
		return err
	}

	return nil
}

func UnpackTarGz(tarGzPath string, destDir string) error {
	file, err := os.Open(tarGzPath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		target := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, os.ModePerm); err != nil {
				return err
			}
		case tar.TypeReg:
			outFile, err := os.Create(target)
			if err != nil {
				return err
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, tr); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported type: %c in %s", header.Typeflag, header.Name)
		}
	}
	return nil
}

func CreatePipeline(modelsPath string, device string) Model {
	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)

	var pipeline *C.ov_genai_llm_pipeline

	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	// C.ov_genai_llm_pipeline_create(cModelsPath, cDevice, &pipeline)
	if device == "NPU" {
		C.ov_genai_llm_pipeline_create_npu_output_2048(cModelsPath, cDevice, &pipeline)
	} else {
		C.ov_genai_llm_pipeline_create_cgo(cModelsPath, cDevice, &pipeline)
	}
	return pipeline
}

func CreateVlmPipeline(modelsPath string, device string) VlmModel {
	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)

	var pipeline *C.ov_genai_vlm_pipeline

	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	C.ov_genai_vlm_pipeline_create_cgo(cModelsPath, cDevice, &pipeline)
	return pipeline
}

func PrintGenaiMetrics(metrics *C.ov_genai_perf_metrics) {

	log.Printf("Genai Metrics info:")

	var loadtime C.float
	C.ov_genai_perf_metrics_get_load_time(metrics, &loadtime)
	log.Printf("Load time: %.2f", loadtime)

	var gen_mean C.float
	var gen_std C.float
	C.ov_genai_perf_metrics_get_inference_duration(metrics, &gen_mean, &gen_std)
	log.Printf("Generate time: %.2f ± %.2f ms\n", gen_mean, gen_std)

	var token_mean C.float
	var token_std C.float
	C.ov_genai_perf_metrics_get_tokenization_duration(metrics, &token_mean, &token_std)
	log.Printf("Tokenization time: %.2f ± %.2f ms\n", token_mean, token_std)

	var detoken_mean C.float
	var detoken_std C.float
	C.ov_genai_perf_metrics_get_detokenization_duration(metrics, &detoken_mean, &detoken_std)
	log.Printf("Detokenization time: %.2f ± %.2f ms\n", detoken_mean, detoken_std)

	var ttft_mean C.float
	var ttft_std C.float
	C.ov_genai_perf_metrics_get_ttft(metrics, &ttft_mean, &ttft_std)
	log.Printf("TTFT: %.2f ± %.2f ms\n", ttft_mean, ttft_std)

	var tpot_mean C.float
	var tpot_std C.float
	C.ov_genai_perf_metrics_get_tpot(metrics, &tpot_mean, &tpot_std)
	log.Printf("TPOT: %.2f ± %.2f ms/token\n", tpot_mean, tpot_std)

	var num_generation_tokens C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &num_generation_tokens)
	log.Printf("Num of generation tokens: %d\n", num_generation_tokens)

	var num_input_tokens C.size_t
	C.ov_genai_perf_metrics_get_num_input_tokens(metrics, &num_input_tokens)
	log.Printf("Num of input tokens: %d\n", num_input_tokens)

	var tput_mean C.float
	var tput_std C.float
	C.ov_genai_perf_metrics_get_throughput(metrics, &tput_mean, &tput_std)
	log.Printf("Throughput: %.2f ± %.2f tokens/s\n", tput_mean, tput_std)
}

// applyPerfMetricsToSequence copies OpenVINO-reported token counts onto the sequence
// so the genairunner HTTP layer can populate PromptN / PredictedN for Ollama usage APIs.
func applyPerfMetricsToSequence(metrics *C.ov_genai_perf_metrics, seq *Sequence) {
	if metrics == nil || seq == nil {
		return
	}
	var numGen C.size_t
	var numIn C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &numGen)
	C.ov_genai_perf_metrics_get_num_input_tokens(metrics, &numIn)
	seq.SetOpenVINOTokenCounts(int(numIn), int(numGen))
	seq.SetOVPerfApplied(true)
}

func SetSamplingParams(samplingparameters *SamplingParams) *C.ov_genai_generation_config {
	var cConfig *C.ov_genai_generation_config
	C.ov_genai_generation_config_create(&cConfig)

	log.Printf("Sampling Parameters - Temperature: %.2f, TopP: %.2f, TopK: %d, RepeatPenalty: %.2f",
		samplingparameters.Temp,
		samplingparameters.TopP,
		samplingparameters.TopK,
		samplingparameters.RepeatPenalty)

	C.ov_genai_generation_config_set_max_new_tokens(cConfig, C.size_t(samplingparameters.MaxNewToken))
	C.ov_genai_generation_config_set_temperature(cConfig, C.float(samplingparameters.Temp))
	C.ov_genai_generation_config_set_top_p(cConfig, C.float(samplingparameters.TopP))
	// C.ov_genai_generation_config_set_top_k(cConfig, C.size_t(samplingparameters.TopK))

	if samplingparameters.StopIds != nil {
		cStopArray := C.malloc(C.size_t(len(samplingparameters.StopIds)) * C.size_t(unsafe.Sizeof(C.int64_t(0))))
		defer C.free(cStopArray)

		var stopArray []int64

		for i := 0; i < int(len(samplingparameters.StopIds)); i++ {
			stop_id, _ := strconv.ParseInt(samplingparameters.StopIds[i], 10, 64)
			stopArray = append(stopArray, stop_id)
		}

		for i, v := range stopArray {
			*(*C.int64_t)(unsafe.Pointer(uintptr(cStopArray) + uintptr(i)*unsafe.Sizeof(C.int64_t(0)))) = C.int64_t(v)
		}
		C.ov_genai_generation_config_set_stop_token_ids(cConfig, (*C.int64_t)(cStopArray), C.size_t(len(stopArray)))
	}

	C.ov_genai_generation_config_set_repetition_penalty(cConfig, C.float(samplingparameters.RepeatPenalty))

	if samplingparameters.StopString != nil {
		cStopStrings := C.malloc(C.size_t(len(samplingparameters.StopString)) * C.size_t(unsafe.Sizeof(uintptr(0))))
		defer C.free(cStopStrings)
		for i, s := range samplingparameters.StopString {
			cStr := C.CString(s)               // Convert Go string to C string
			defer C.free(unsafe.Pointer(cStr)) // Free each C string after use
			*(*uintptr)(unsafe.Pointer(uintptr(cStopStrings) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(cStr))
		}
		C.ov_genai_generation_config_set_stop_strings(cConfig, (**C.char)(cStopStrings), C.size_t(len(samplingparameters.StopString)))
	}

	return cConfig
}

func GenerateTextWithMetrics(pipeline *C.ov_genai_llm_pipeline, input string, samplingparameters *SamplingParams, seq *Sequence) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	// 构建 chat_history
	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	// 将 prompt 构建为 user message JSON 并加入 history
	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to push message to chat history: %d", chatStatus)
		return ""
	}

	// 如果有 tools，通过 chat_history 设置
	tools := seq.GetTools()
	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("failed to marshal tools: %v", err)
		} else {
			// log.Printf("setting tools on chat_history: %s", string(toolsJSON))
			log.Printf("setting tools on chat_history: %d tool(s)", len(tools))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus = C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("failed to set tools on chat history: %d", chatStatus)
				} else {
					log.Printf("tools set on chat_history successfully")
				}
			}
		}
	}

	// Set enable_thinking via extra_context on chat_history
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("failed to set extra_context on chat history: %d", chatStatus)
		} else {
			log.Printf("enable_thinking=%t set on chat_history extra_context", samplingparameters.EnableThinking)
		}
	}

	// 使用 generate_with_history 替代 generate
	C.ov_genai_llm_pipeline_generate_with_history(pipeline,
		chatHistory,
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("genai decoded output (final cOutput):\n%s", out)
	return out
}

// addMessageToChatHistory marshals a message map to JSON and pushes it into the C chat_history.
func addMessageToChatHistory(chatHistory *C.ov_genai_chat_history, msg map[string]any) error {
	msgJSON, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	cMsgJSON := C.CString(string(msgJSON))
	defer C.free(unsafe.Pointer(cMsgJSON))

	var container *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&container, cMsgJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		return fmt.Errorf("failed to create json container: %d", jsonStatus)
	}
	defer C.ov_genai_json_container_free(container)

	chatStatus := C.ov_genai_chat_history_push_back(chatHistory, container)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		return fmt.Errorf("failed to push message to chat history: %d", chatStatus)
	}
	return nil
}

// GenerateWithChatHistory builds a full chat_history from []api.Message,
// sets tools and extra_context, then calls generate_with_history.
// This follows the OpenVINO GenAI architecture: tool responses are added
// as separate messages with role "tool" into chat_history, and the chat
// template handles all formatting internally.
func GenerateWithChatHistory(pipeline *C.ov_genai_llm_pipeline, messages []api.Message,
	tools []api.Tool, samplingparameters *SamplingParams, seq *Sequence) string {

	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	for _, msg := range messages {
		msgMap := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var toolCallsData []map[string]any
			for _, tc := range msg.ToolCalls {
				tcMap := map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments.ToMap(),
					},
				}
				toolCallsData = append(toolCallsData, tcMap)
			}
			msgMap["tool_calls"] = toolCallsData
		}

		if msg.Role == "tool" {
			if msg.ToolCallID != "" {
				msgMap["tool_call_id"] = msg.ToolCallID
			}
			if msg.ToolName != "" {
				msgMap["name"] = msg.ToolName
			}
		}

		if err := addMessageToChatHistory(chatHistory, msgMap); err != nil {
			log.Printf("failed to add %s message to chat history: %v", msg.Role, err)
			return ""
		}
	}

	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("failed to marshal tools: %v", err)
		} else {
			// log.Printf("setting tools on chat_history: %s", string(toolsJSON))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus := C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	log.Printf("enable_thinking: %t", samplingparameters.EnableThinking)
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	C.ov_genai_llm_pipeline_generate_with_history(pipeline,
		chatHistory,
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("genai decoded output (final cOutput):\n%s", out)
	return out
}

type ImageInfo struct {
	Width    int
	Height   int
	Format   string
	Channels int
}

func getImageInfo(imageData []byte) (*ImageInfo, error) {
	reader := bytes.NewReader(imageData)

	config, format, err := image.DecodeConfig(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image config: %v", err)
	}

	channels := 3

	return &ImageInfo{
		Width:    config.Width,
		Height:   config.Height,
		Format:   format,
		Channels: channels,
	}, nil
}

func decodeImageToRGB(imageData []byte) ([]byte, error) {
	reader := bytes.NewReader(imageData)

	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	rgbData := make([]byte, width*height*3)
	idx := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			rgbData[idx] = byte(r >> 8)   // R
			rgbData[idx+1] = byte(g >> 8) // G
			rgbData[idx+2] = byte(b >> 8) // B
			idx += 3
		}
	}

	return rgbData, nil
}

func createTensorFromImageData(imageData []byte) (*C.ov_tensor_t, error) {
	if len(imageData) == 0 {
		return nil, fmt.Errorf("empty image data")
	}

	imgInfo, err := getImageInfo(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to get image info: %v", err)
	}

	height := int64(imgInfo.Height)
	width := int64(imgInfo.Width)
	channels := int64(imgInfo.Channels)

	decodedData, err := decodeImageToRGB(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	expectedSize := height * width * channels
	if int64(len(decodedData)) < expectedSize {
		return nil, fmt.Errorf("decoded image data size mismatch: expected %d, got %d", expectedSize, len(decodedData))
	}

	dims := []int64{1, height, width, channels}

	// create ov_shape_t
	var shape C.ov_shape_t
	status := C.ov_shape_create(
		C.int64_t(4),                           // rank
		(*C.int64_t)(unsafe.Pointer(&dims[0])), // dims
		&shape,
	)
	if status != C.OK {
		return nil, fmt.Errorf("failed to create shape: %d", status)
	}
	defer C.ov_shape_free(&shape)

	var tensor *C.ov_tensor_t
	status = C.ov_tensor_create_from_host_ptr(
		C.U8,                            // element type (uint8)
		shape,                           // shape
		unsafe.Pointer(&decodedData[0]), // host pointer to decoded data
		&tensor,
	)
	if status != C.OK {
		return nil, fmt.Errorf("failed to create tensor: %d", status)
	}

	return tensor, nil
}

func vlmImageTensors(images []llamaserver.ImageData) ([]*C.ov_tensor_t, **C.ov_tensor_t) {
	var rgbs []*C.ov_tensor_t
	for i, imageData := range images {
		tensor, err := createTensorFromImageData(imageData.Data)
		if err != nil {
			log.Printf("vlm: error creating tensor for image %d: %v", i, err)
			continue
		}
		rgbs = append(rgbs, tensor)
	}
	var cRgbs **C.ov_tensor_t
	if len(rgbs) > 0 {
		cRgbs = (**C.ov_tensor_t)(unsafe.Pointer(&rgbs[0]))
	}
	return rgbs, cRgbs
}

// VlmGenerateWithChatHistory builds chat_history from messages (same structure as LLM),
// sets tools and enable_thinking on history, then calls ov_genai_vlm_pipeline_generate_with_history.
// rgbs must be images for the last user turn per OpenVINO VLM C API.
func VlmGenerateWithChatHistory(pipeline *C.ov_genai_vlm_pipeline, messages []api.Message,
	images []llamaserver.ImageData, tools []api.Tool, samplingparameters *SamplingParams, seq *Sequence) string {

	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	for _, msg := range messages {
		msgMap := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var toolCallsData []map[string]any
			for _, tc := range msg.ToolCalls {
				tcMap := map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments.ToMap(),
					},
				}
				toolCallsData = append(toolCallsData, tcMap)
			}
			msgMap["tool_calls"] = toolCallsData
		}

		if msg.Role == "tool" {
			if msg.ToolCallID != "" {
				msgMap["tool_call_id"] = msg.ToolCallID
			}
			if msg.ToolName != "" {
				msgMap["name"] = msg.ToolName
			}
		}

		if err := addMessageToChatHistory(chatHistory, msgMap); err != nil {
			log.Printf("vlm: failed to add %s message to chat history: %v", msg.Role, err)
			return ""
		}
	}

	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("vlm: failed to marshal tools: %v", err)
		} else {
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus := C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("vlm: failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("vlm: failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	log.Printf("vlm enable_thinking: %t", samplingparameters.EnableThinking)
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)
	log.Printf("vlm generate_with_history: %d messages, %d image tensors", len(messages), len(rgbs))

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_vlm_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("vlm genai decoded output (final cOutput):\n%s", out)
	return out
}

// VlmGenerateTextWithMetrics uses a single user message in chat_history (LLM GenerateTextWithMetrics parity),
// including tools and enable_thinking, then ov_genai_vlm_pipeline_generate_with_history.
func VlmGenerateTextWithMetrics(pipeline *C.ov_genai_vlm_pipeline, input string, images []llamaserver.ImageData, samplingparameters *SamplingParams, seq *Sequence) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("vlm: failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to push message to chat history: %d", chatStatus)
		return ""
	}

	tools := seq.GetTools()
	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("vlm: failed to marshal tools: %v", err)
		} else {
			log.Printf("vlm: setting tools on chat_history: %d tool(s)", len(tools))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus = C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("vlm: failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("vlm: failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		} else {
			log.Printf("vlm: enable_thinking=%t set on chat_history extra_context", samplingparameters.EnableThinking)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)
	log.Printf("vlm: prompt_len=%d images=%d (successful tensors=%d)", len(input), len(images), len(rgbs))

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_vlm_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("vlm result: %s", out)
	return out
}

func VlmGenerateText(pipeline *C.ov_genai_vlm_pipeline, input string, images []llamaserver.ImageData, samplingparameters *SamplingParams) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("vlm: failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to push message to chat history: %d", chatStatus)
		return ""
	}

	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		nil,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)
	return C.GoString((*C.char)(cOutput))
}
func GenerateText(pipeline *C.ov_genai_llm_pipeline, input string, streamer_callback C.streamer_callback) string {
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	output_size := C.size_t(0)

	var result *C.ov_genai_decoded_results

	C.ov_genai_llm_pipeline_start_chat(pipeline)
	C.ov_genai_llm_pipeline_generate(pipeline, cInput, (*C.ov_genai_generation_config)(nil), &streamer_callback, &result)
	C.ov_genai_llm_pipeline_finish_chat(pipeline)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)
	return C.GoString((*C.char)(cOutput))
}

func FreeModel(model Model) {
	C.ov_genai_llm_pipeline_free(model)
}

func GetGenaiAvailableDevices() []map[string]string {
	var core *(C.ov_core_t)
	defer C.ov_core_free(core)

	C.ov_core_create(&core)

	var devices C.ov_available_devices_t
	C.ov_core_get_available_devices(core, &devices)

	// goDevice := make(map[string]map[string][]string)
	var goDevices []map[string]string

	cString := devices.devices
	for i := 0; i < int(devices.size); i++ {
		var version_list C.ov_core_version_list_t

		C.ov_core_get_versions_by_device_name(core, (*C.char)(*cString), &version_list)

		cversion_list := version_list
		for i := 0; i < int(version_list.size); i++ {
			version := (*C.ov_core_version_t)(unsafe.Pointer(uintptr(unsafe.Pointer(cversion_list.versions)) + uintptr(i)*unsafe.Sizeof(*cversion_list.versions)))
			deviceName := C.GoString(version.device_name) // 获取 device_name
			buildNumber := C.GoString(version.version.buildNumber)
			description := C.GoString(version.version.description)

			item := map[string]string{"device_name": deviceName, "buildNumber": buildNumber, "description": description}
			goDevices = append(goDevices, item)
		}

		cString = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cString)) + unsafe.Sizeof(*cString)))

	}
	defer C.ov_available_devices_free(&devices)

	return goDevices
}

func GetOvVersion() {
	var ov_version C.ov_version_t
	C.ov_get_openvino_version(&ov_version)
	log.Printf("---- OpenVINO INFO----\n")
	log.Printf("Description : %s \n", C.GoString(ov_version.description))
	log.Printf("Build number: %s \n", C.GoString(ov_version.buildNumber))

	defer C.ov_version_free(&ov_version)
}

//export goCallbackBridge
func goCallbackBridge(args *C.char, gen_result unsafe.Pointer) C.int {
	if args != nil {
		// 将 unsafe.Pointer 转换回结构体指针
		handle := cgo.Handle(uintptr(gen_result))
		result := handle.Value().(*Sequence)

		// 将 C 字符串转换为 Go 字符串并追加到切片中
		goStr := C.GoString(args)
		result.AppendPendingResponse(goStr)

		// fmt.Printf("%s", goStr)
		// os.Stdout.Sync()
		FlushPending((*Sequence)(result))
		return C.int(C.OV_GENAI_STREAMING_STATUS_RUNNING_COMPAT)
	} else {
		fmt.Println("Callback executed with NULL message!")
		return C.int(C.OV_GENAI_STREAMING_STATUS_STOP_COMPAT)
	}
}

func FreeVlmModel(pipeline *C.ov_genai_vlm_pipeline) {
	C.ov_genai_vlm_pipeline_free(pipeline)
}

// func main() {
// 	// 使用 createPipeline 函数创建 pipeline
// 	// log.Printf(GetGenaiAvailableDevices()[0]["device_name"])
// 	GetOvVersion()
// 	pipeline := CreatePipeline("/home/hongbo/ollama_model/TinyLlama-1.1B-Chat-v1.0-int4-ov", "CPU")
// 	if pipeline == nil {
// 		fmt.Println("创建 pipeline 失败")
// 		return
// 	}
// 	fmt.Println("成功创建 pipeline")

// 	// 创建 streamer_callback
// 	var streamer_callback C.streamer_callback
// 	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

// 	// 初始化 gena_result 结构体
// 	gena_result := &gen_result_struct{
// 		pendingResponses: make([]string, 0),
// 	}

// 	// 使用 runtime.Pinner 固定指针
// 	var pinner runtime.Pinner
// 	pinner.Pin(gena_result)
// 	defer pinner.Unpin()

// 	streamer_callback.args = unsafe.Pointer(gena_result)

// 	result := GenerateText(pipeline, "who you are?", streamer_callback)
// 	fmt.Println("生成的结果 result:", result)

// 	if gena_result != nil && len(gena_result.pendingResponses) > 0 {
// 		fmt.Println("生成的结果 gen_result:", gena_result.pendingResponses)
// 	} else {
// 		fmt.Println("生成的结果 gen_result: 无响应")
// 	}
// 	// 释放 pipeline 资源
// 	FreeModel(pipeline)
// }
