package util

import (
	"errors"
	"math"
)

func IndentExpand(indent string, growth int) string {
	indentByte := []byte(indent)
	out := make([]byte, 0, len(indent)*growth)
	for i := 0; i < growth; i++ {
		out = append(out, indentByte...)
	}
	return string(out)
}

func SliceMap(arr []float64, lambda func(float64) (float64, error)) ([]float64, error) {
	var err error
	for i, v := range arr {
		arr[i], err = lambda(v)
		if err != nil {
			return nil, err
		}
	}
	return arr, nil
}

var ErrNegativeDataWithLog = errors.New("cannot use log transformation with negative data")

func LogTranformSeries(arr []float64) ([]float64, error) {
	return SliceMap(
		arr,
		func(y float64) (float64, error) {
			if y < 0 {
				return 0.0, ErrNegativeDataWithLog
			}
			return math.Log1p(y), nil
		},
	)
}
