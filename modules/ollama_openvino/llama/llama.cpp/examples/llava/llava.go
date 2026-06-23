package llava

// #cgo !windows CXXFLAGS: -std=c++11
// #cgo windows CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -I${SRCDIR}/../../include -I${SRCDIR}/../../common
// #cgo CPPFLAGS: -I${SRCDIR}/../../../../ml/backend/ggml/ggml/include
import "C"
