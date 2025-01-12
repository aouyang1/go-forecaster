// floatsunrolled is inspired by the SIMD blog post
// https://github.com/camdencheek/simd_blog/blob/main/main.go
package floatsunrolled

import (
	"errors"
	"fmt"
)

const UnrollBatch = 4

var (
	ErrSliceLengthMismatch       = errors.New("slices must have equal lengths")
	ErrSliceMul                  = fmt.Errorf("slice length must be multiple of %d", UnrollBatch)
	ErrOutputSliceLengthMismatch = errors.New("output slice length not the same as input")
)

func Dot(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(ErrSliceLengthMismatch)
	}

	if len(a)%UnrollBatch != 0 {
		panic(ErrSliceMul)
	}

	var sum float64
	for i := 0; i < len(a); i += UnrollBatch {
		aTmp := a[i : i+UnrollBatch : i+UnrollBatch]
		bTmp := b[i : i+UnrollBatch : i+UnrollBatch]
		s0 := aTmp[0] * bTmp[0]
		s1 := aTmp[1] * bTmp[1]
		s2 := aTmp[2] * bTmp[2]
		s3 := aTmp[3] * bTmp[3]
		sum += s0 + s1 + s2 + s3
	}
	return sum
}

func Add(dst, s []float64) []float64 {
	if len(dst) != len(s) {
		panic(ErrSliceLengthMismatch)
	}

	if len(s)%UnrollBatch != 0 {
		panic(ErrSliceMul)
	}

	for i := 0; i < len(s); i += UnrollBatch {
		dstTmp := dst[i : i+UnrollBatch : i+UnrollBatch]
		sTmp := s[i : i+UnrollBatch : i+UnrollBatch]
		dstTmp[0] += sTmp[0]
		dstTmp[1] += sTmp[1]
		dstTmp[2] += sTmp[2]
		dstTmp[3] += sTmp[3]
	}
	return dst
}

func SubTo(dst, s, t []float64) []float64 {
	if len(s) != len(t) {
		panic(ErrSliceLengthMismatch)
	}

	if len(s)%UnrollBatch != 0 {
		panic(ErrSliceMul)
	}

	if dst == nil {
		dst = make([]float64, len(s))
	} else if len(dst) != len(s) {
		panic(ErrOutputSliceLengthMismatch)
	}

	for i := 0; i < len(s); i += UnrollBatch {
		dstTmp := dst[i : i+UnrollBatch : i+UnrollBatch]
		sTmp := s[i : i+UnrollBatch : i+UnrollBatch]
		tTmp := t[i : i+UnrollBatch : i+UnrollBatch]
		dstTmp[0] = sTmp[0] - tTmp[0]
		dstTmp[1] = sTmp[1] - tTmp[1]
		dstTmp[2] = sTmp[2] - tTmp[2]
		dstTmp[3] = sTmp[3] - tTmp[3]
	}

	return dst
}

func ScaleTo(dst []float64, c float64, s []float64) []float64 {
	if len(s)%UnrollBatch != 0 {
		panic(ErrSliceMul)
	}

	if dst == nil {
		dst = make([]float64, len(s))
	} else if len(dst) != len(s) {
		panic(ErrOutputSliceLengthMismatch)
	}

	for i := 0; i < len(s); i += UnrollBatch {
		dstTmp := dst[i : i+UnrollBatch : i+UnrollBatch]
		sTmp := s[i : i+UnrollBatch : i+UnrollBatch]
		dstTmp[0] = c * sTmp[0]
		dstTmp[1] = c * sTmp[1]
		dstTmp[2] = c * sTmp[2]
		dstTmp[3] = c * sTmp[3]
	}

	return dst
}
