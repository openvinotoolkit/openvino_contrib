package genai

/*
#cgo !windows CFLAGS: -std=c11
#cgo windows CFLAGS: -std=c11
#cgo !windows CXXFLAGS: -std=c++17
#cgo windows CXXFLAGS: -std=c++17

#cgo !windows LDFLAGS: -lopenvino_genai -lopenvino -lopenvino_c -lopenvino_genai_c
#cgo windows LDFLAGS: -lopenvino_genai -lopenvino -lopenvino_c -lopenvino_genai_c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openvino/genai/c/llm_pipeline.h"
#include "openvino/genai/c/visibility.h"

#include "openvino/c/openvino.h"
#include "openvino/c/ov_common.h"
#include <stdbool.h>

typedef int (*callback_function)(const char*, void*);

extern int goCallbackBridge(char* input, void* ptr);

static ov_status_e ov_genai_llm_pipeline_create_npu_output_2048(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, "NPU", 4, pipe, "MAX_PROMPT_LEN", "2048", "MIN_RESPONSE_LEN", "2048");
}

static ov_status_e ov_genai_llm_pipeline_create_cgo(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, device, 0, pipe);
}

static ov_status_e ov_genai_llm_pipeline_generate_cgo(ov_genai_llm_pipeline* pipeline,
                                                                  const char* input,
                                                                  ov_genai_generation_config* config,
                                                                  streamer_callback* callback,
                                                                  ov_genai_decoded_results** result) {
	ov_genai_llm_pipeline_start_chat(pipeline);
	ov_status_e status = ov_genai_llm_pipeline_generate(pipeline, input, config, callback, result);
	ov_genai_llm_pipeline_finish_chat(pipeline);
	return status;
}

static void ov_genai_generation_config_free_cgo(ov_genai_generation_config* config) {
	if (config) ov_genai_generation_config_free(config);
}

*/
import "C"

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/cgo"
	"strconv"
	"strings"
	"unsafe"
)

type SamplingParams struct {
	TopK          int
	TopP          float32
	Temp          float32
	MaxNewToken   int
	RepeatPenalty float32
	StopString    []string
	StopIds       []string
	RepeatLastN   int
}

type Model *C.ov_genai_llm_pipeline

func ovStatusError(op string, status C.ov_status_e) error {
	if int(status) == 0 {
		return nil
	}

	return fmt.Errorf("%s failed with OpenVINO status code %d", op, int(status))
}

func normalizeDeviceName(device string) string {
	return strings.ToUpper(strings.TrimSpace(device))
}

func devicePluginName(device string) string {
	normalized := normalizeDeviceName(device)
	if idx := strings.Index(normalized, "."); idx >= 0 {
		return normalized[:idx]
	}

	return normalized
}

func getAvailableDeviceNames() ([]string, error) {
	var core *C.ov_core_t
	status := C.ov_core_create(&core)
	if err := ovStatusError("creating OpenVINO core", status); err != nil {
		return nil, err
	}
	if core == nil {
		return nil, fmt.Errorf("creating OpenVINO core returned a nil core")
	}
	defer C.ov_core_free(core)

	var devices C.ov_available_devices_t
	status = C.ov_core_get_available_devices(core, &devices)
	if err := ovStatusError("getting OpenVINO available devices", status); err != nil {
		return nil, err
	}
	defer C.ov_available_devices_free(&devices)

	available := make([]string, 0, int(devices.size))
	cString := devices.devices
	for i := 0; i < int(devices.size); i++ {
		available = append(available, C.GoString(*cString))
		cString = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cString)) + unsafe.Sizeof(*cString)))
	}

	return available, nil
}

func hasAvailableDevice(availableDevices []string, requested string) bool {
	requested = normalizeDeviceName(requested)
	if requested == "" {
		return false
	}

	requestsSpecificID := strings.Contains(requested, ".")
	for _, availableDevice := range availableDevices {
		normalizedAvailable := normalizeDeviceName(availableDevice)
		if normalizedAvailable == requested {
			return true
		}
		if !requestsSpecificID && devicePluginName(normalizedAvailable) == requested {
			return true
		}
	}

	return false
}

func fallbackDevice(availableDevices []string) string {
	switch {
	case hasAvailableDevice(availableDevices, "CPU"):
		return "CPU"
	case hasAvailableDevice(availableDevices, "GPU"):
		return "GPU"
	case len(availableDevices) > 0:
		return normalizeDeviceName(availableDevices[0])
	default:
		return ""
	}
}

func defaultDevice(availableDevices []string) string {
	switch {
	case hasAvailableDevice(availableDevices, "GPU"):
		return "GPU"
	case hasAvailableDevice(availableDevices, "CPU"):
		return "CPU"
	case len(availableDevices) > 0:
		return normalizeDeviceName(availableDevices[0])
	default:
		return ""
	}
}

func ResolveDeviceOrFallback(device string) (string, error) {
	availableDevices, err := getAvailableDeviceNames()
	if err != nil {
		return "", err
	}
	if len(availableDevices) == 0 {
		return "", fmt.Errorf("OpenVINO reported no available devices")
	}

	requestedDevice := normalizeDeviceName(device)
	if requestedDevice == "" {
		resolved := defaultDevice(availableDevices)
		if resolved == "" {
			return "", fmt.Errorf("OpenVINO reported no default device (available: %s)", strings.Join(availableDevices, ", "))
		}

		return resolved, nil
	}

	if hasAvailableDevice(availableDevices, requestedDevice) {
		return requestedDevice, nil
	}

	resolved := fallbackDevice(availableDevices)
	if resolved == "" {
		return "", fmt.Errorf("OpenVINO device %q is not available and no fallback device was found (available: %s)", requestedDevice, strings.Join(availableDevices, ", "))
	}

	log.Printf("OpenVINO device %q is not available (available: %s); falling back to %q", requestedDevice, strings.Join(availableDevices, ", "), resolved)
	if requestedDevice == "NPU" && resolved == "CPU" {
		fmt.Println("NPU is not available, falling back to CPU")
	}
	return resolved, nil
}

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

func CreatePipeline(modelsPath string, device string) (*C.ov_genai_llm_pipeline, error) {
	resolvedDevice, err := ResolveDeviceOrFallback(device)
	if err != nil {
		return nil, fmt.Errorf("resolving OpenVINO device %q: %w", device, err)
	}

	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(resolvedDevice)

	var pipeline *C.ov_genai_llm_pipeline
	var status C.ov_status_e

	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	if resolvedDevice == "NPU" {
		status = C.ov_genai_llm_pipeline_create_npu_output_2048(cModelsPath, cDevice, &pipeline)
	} else {
		status = C.ov_genai_llm_pipeline_create_cgo(cModelsPath, cDevice, &pipeline)
	}

	if err := ovStatusError("creating OpenVINO GenAI pipeline", status); err != nil {
		return nil, fmt.Errorf("%w for %q on %s", err, modelsPath, resolvedDevice)
	}

	if pipeline == nil {
		return nil, fmt.Errorf("creating OpenVINO GenAI pipeline returned a nil pipeline for %q on %s", modelsPath, resolvedDevice)
	}

	return pipeline, nil
}

func PrintGenaiMetrics(metrics *C.ov_genai_perf_metrics) {

	log.Printf("Genai Metrics info:")

	var loadtime C.float
	C.ov_genai_perf_metrics_get_load_time(metrics, &loadtime)
	log.Printf("Load time: %.2f", loadtime)

	var gen_mean C.float
	var gen_std C.float
	C.ov_genai_perf_metrics_get_inference_duration(metrics, &gen_mean, &gen_std)
	log.Printf("Generate time: %.2f +/- %.2f ms\n", gen_mean, gen_std)

	var token_mean C.float
	var token_std C.float
	C.ov_genai_perf_metrics_get_tokenization_duration(metrics, &token_mean, &token_std)
	log.Printf("Tokenization time: %.2f +/- %.2f ms\n", token_mean, token_std)

	var detoken_mean C.float
	var detoken_std C.float
	C.ov_genai_perf_metrics_get_detokenization_duration(metrics, &detoken_mean, &detoken_std)
	log.Printf("Detokenization time: %.2f +/- %.2f ms\n", detoken_mean, detoken_std)

	var ttft_mean C.float
	var ttft_std C.float
	C.ov_genai_perf_metrics_get_ttft(metrics, &ttft_mean, &ttft_std)
	log.Printf("TTFT: %.2f +/- %.2f ms\n", ttft_mean, ttft_std)

	var tpot_mean C.float
	var tpot_std C.float
	C.ov_genai_perf_metrics_get_tpot(metrics, &tpot_mean, &tpot_std)
	log.Printf("TPOT: %.2f +/- %.2f ms/token\n", tpot_mean, tpot_std)

	var num_generation_tokens C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &num_generation_tokens)
	log.Printf("Num of generation tokens: %d\n", num_generation_tokens)

	var tput_mean C.float
	var tput_std C.float
	C.ov_genai_perf_metrics_get_throughput(metrics, &tput_mean, &tput_std)
	log.Printf("Throughput: %.2f +/- %.2f tokens/s\n", tput_mean, tput_std)
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

func GenerateTextWithMetrics(pipeline *C.ov_genai_llm_pipeline, input string, samplingparameters *SamplingParams, seq *Sequence) (string, error) {
	fmt.Println("GENERATION STARTED - input length:", len(input))
	if pipeline == nil {
		return "", fmt.Errorf("OpenVINO pipeline is not initialized")
	}

	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	cConfig := SetSamplingParams(samplingparameters)
	defer C.ov_genai_generation_config_free_cgo(cConfig)
	var result *C.ov_genai_decoded_results

	output_size := C.size_t(0)

	handle := cgo.NewHandle(seq)
	defer handle.Delete()

	cHandle := C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	defer C.free(cHandle)
	*(*C.uintptr_t)(cHandle) = C.uintptr_t(handle)

	// Create streamer_callback
	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))
	streamer_callback.args = cHandle

	status := C.ov_genai_llm_pipeline_generate_cgo(
		pipeline,
		cInput,
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result,
	)
	if err := ovStatusError("generating text with OpenVINO GenAI", status); err != nil {
		return "", err
	}

	if result == nil {
		return "", fmt.Errorf("OpenVINO GenAI generate returned no decoded result")
	}

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	defer C.ov_genai_decoded_results_free(result)

	// Declare output first, then use it
	output := C.GoString((*C.char)(cOutput))

	var generatedTokens C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &generatedTokens)

	if strings.TrimSpace(output) == "" && generatedTokens == 0 {
		log.Println("Warning: Empty output from OpenVINO")
	}

	if strings.TrimSpace(output) == "" && generatedTokens == 0 && strings.TrimSpace(input) != "" {
		return "", fmt.Errorf("OpenVINO GenAI returned an empty completion with 0 generated tokens")
	}

	return output, nil
}

func GenerateText(pipeline *C.ov_genai_llm_pipeline, input string, streamer_callback C.streamer_callback) (string, error) {
	if pipeline == nil {
		return "", fmt.Errorf("OpenVINO pipeline is not initialized")
	}

	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	output_size := C.size_t(0)

	var result *C.ov_genai_decoded_results

	status := C.ov_genai_llm_pipeline_generate_cgo(pipeline, cInput, (*C.ov_genai_generation_config)(nil), &streamer_callback, &result)
	if err := ovStatusError("generating text with OpenVINO GenAI", status); err != nil {
		return "", err
	}
	if result == nil {
		return "", fmt.Errorf("OpenVINO GenAI generate returned no decoded result")
	}

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)
	return C.GoString((*C.char)(cOutput)), nil
}

func FreeModel(model Model) {
	C.ov_genai_llm_pipeline_free(model)
}

func GetGenaiAvailableDevices() []map[string]string {
	var core *C.ov_core_t
	status := C.ov_core_create(&core)
	if err := ovStatusError("creating OpenVINO core", status); err != nil {
		log.Printf("Failed to enumerate OpenVINO devices: %v", err)
		return nil
	}
	if core == nil {
		log.Printf("Failed to enumerate OpenVINO devices: creating OpenVINO core returned a nil core")
		return nil
	}
	defer C.ov_core_free(core)

	var devices C.ov_available_devices_t
	status = C.ov_core_get_available_devices(core, &devices)
	if err := ovStatusError("getting OpenVINO available devices", status); err != nil {
		log.Printf("Failed to enumerate OpenVINO devices: %v", err)
		return nil
	}
	defer C.ov_available_devices_free(&devices)

	var goDevices []map[string]string

	cString := devices.devices
	for i := 0; i < int(devices.size); i++ {
		availableDeviceName := C.GoString(*cString)
		var versionList C.ov_core_version_list_t

		status = C.ov_core_get_versions_by_device_name(core, (*C.char)(*cString), &versionList)
		if err := ovStatusError("getting OpenVINO device version", status); err != nil {
			goDevices = append(goDevices, map[string]string{"device_name": availableDeviceName})
			cString = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cString)) + unsafe.Sizeof(*cString)))
			continue
		}

		if versionList.size == 0 {
			goDevices = append(goDevices, map[string]string{"device_name": availableDeviceName})
		}

		for j := 0; j < int(versionList.size); j++ {
			version := (*C.ov_core_version_t)(unsafe.Pointer(uintptr(unsafe.Pointer(versionList.versions)) + uintptr(j)*unsafe.Sizeof(*versionList.versions)))
			buildNumber := C.GoString(version.version.buildNumber)
			description := C.GoString(version.version.description)

			item := map[string]string{"device_name": availableDeviceName, "buildNumber": buildNumber, "description": description}
			goDevices = append(goDevices, item)
		}

		cString = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cString)) + unsafe.Sizeof(*cString)))
	}

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
		if gen_result == nil {
			return C.OV_GENAI_STREAMING_STATUS_STOP
		}

		handle := cgo.Handle(*(*C.uintptr_t)(gen_result))
		result, ok := handle.Value().(*Sequence)
		if !ok || result == nil {
			return C.OV_GENAI_STREAMING_STATUS_STOP
		}

		goStr := C.GoString(args)
		result.AppendPendingResponse(goStr)

		FlushPending(result)
		return C.OV_GENAI_STREAMING_STATUS_RUNNING
	} else {
		fmt.Println("Callback executed with NULL message!")
		return C.OV_GENAI_STREAMING_STATUS_STOP
	}
}
