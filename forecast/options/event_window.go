package options

import "gonum.org/v1/gonum/dsp/window"

const (
	WindowBartlettHann    = "bartlett_hann"
	WindowBlackman        = "blackman"
	WindowBlackmanHarris  = "blackman_harris"
	WindowBlackmanNuttall = "blackman_nuttall"
	WindowFlatTop         = "flat_top"
	WindowHamming         = "hamming"
	WindowHann            = "hann"
	WindowLanczos         = "lanczos"
	WindowNuttall         = "nuttall"
	WindowRectangular     = "rectangular"
	WindowSine            = "sine"
	WindowTriangular      = "triangular"
	WindowTukey           = "tukey"
)

var WindowParamTukeyAlpha = 0.95

func WindowFunc(name string) func(seq []float64) []float64 {
	var winFunc func(seq []float64) []float64
	switch name {
	case WindowBartlettHann:
		winFunc = window.BartlettHann
	case WindowBlackman:
		winFunc = window.Blackman
	case WindowBlackmanHarris:
		winFunc = window.BlackmanHarris
	case WindowBlackmanNuttall:
		winFunc = window.BlackmanNuttall
	case WindowFlatTop:
		winFunc = window.FlatTop
	case WindowHamming:
		winFunc = window.Hamming
	case WindowHann:
		winFunc = window.Hann
	case WindowLanczos:
		winFunc = window.Lanczos
	case WindowNuttall:
		winFunc = window.Nuttall
	case WindowRectangular:
		winFunc = window.Rectangular
	case WindowSine:
		winFunc = window.Sine
	case WindowTriangular:
		winFunc = window.Triangular
	case WindowTukey:
		winFunc = func(seq []float64) []float64 {
			return window.Tukey{Alpha: WindowParamTukeyAlpha}.Transform(seq)
		}
	default:
		winFunc = window.Rectangular
	}
	return winFunc
}
