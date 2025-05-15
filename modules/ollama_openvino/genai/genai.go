package genai

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
#include "openvino/genai/c/visibility.h"

#include "openvino/c/openvino.h"
#include "openvino/c/ov_common.h"
#include <stdbool.h>

typedef int (*callback_function)(const char*, void*);

extern int goCallbackBridge(char* input, void* ptr);

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

*/
import "C"

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
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

// type Model struct {
// 	pipe *C.LLMPipelineHandle
// }

type Model *C.ov_genai_llm_pipeline

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

func CreatePipeline(modelsPath string, device string) *C.ov_genai_llm_pipeline {
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

	var tput_mean C.float
	var tput_std C.float
	C.ov_genai_perf_metrics_get_throughput(metrics, &tput_mean, &tput_std)
	log.Printf("Throughput: %.2f ± %.2f tokens/s\n", tput_mean, tput_std)
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
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_decoded_results

	output_size := C.size_t(0)

	// 创建 streamer_callback
	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	streamer_callback.args = unsafe.Pointer(seq)

	C.ov_genai_llm_pipeline_start_chat(pipeline)
	C.ov_genai_llm_pipeline_generate(pipeline, cInput, (*C.ov_genai_generation_config)(cConfig), &streamer_callback, &result)
	C.ov_genai_llm_pipeline_finish_chat(pipeline)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	PrintGenaiMetrics(metrics)

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
		result := (*Sequence)(gen_result)

		// 将 C 字符串转换为 Go 字符串并追加到切片中
		goStr := C.GoString(args)
		result.AppendPendingResponse(goStr)

		// fmt.Printf("%s", goStr)
		// os.Stdout.Sync()
		FlushPending((*Sequence)(result))
		return C.OV_GENAI_STREAMMING_STATUS_RUNNING
	} else {
		fmt.Println("Callback executed with NULL message!")
		return C.OV_GENAI_STREAMMING_STATUS_STOP
	}
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
