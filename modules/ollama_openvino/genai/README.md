go build -o test_vlmrunner.exe test_vlmrunner.go

test_vlmrunner.exe -model "C:\hongbo\test\Qwen2.5-VL-3B-Instruct.tar.gz" -modelname zhaohb_testvl:v1 -device GPU
test_runner.exe -model "C:\hongbo\models\DeepSeek-R1-Distill-Qwen-1.5B-int4-ov.tar.gz" -modelname zhaohb_deepseek:v2  -device GPU