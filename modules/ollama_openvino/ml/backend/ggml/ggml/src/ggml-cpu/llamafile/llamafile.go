package llamafile

// #cgo !windows CXXFLAGS: -std=c++17
// #cgo windows CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include
import "C"
