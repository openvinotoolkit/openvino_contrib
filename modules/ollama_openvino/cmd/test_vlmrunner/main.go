package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/genai"
)

func main() {
	args := append([]string{os.Args[0], "--genai-vlm-engine"}, os.Args[1:]...)
	if err := genai.Execute(args); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
		os.Exit(1)
	}
}
