package genai

import (
	"log"

	"github.com/ollama/ollama/genai/runner"
	"github.com/ollama/ollama/genai/vlmrunner"
)

func Execute(args []string) error {
	if args[0] == "genairunner" {
		args = args[1:]
	}

	var vlmRunner bool
	if args[0] == "--genai-vlm-engine" {
		args = args[1:]
		vlmRunner = true
	}

	log.Printf("args: %+v", args)
	log.Printf("vlmRunner: %v", vlmRunner)

	if vlmRunner {
		return vlmrunner.Execute(args)
	} else {
		return runner.Execute(args)
	}
}
