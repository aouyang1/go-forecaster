package main

import (
	"errors"
	"fmt"
	"math"
)

var ErrResLenMismatch = errors.New("predicted and actual have different lengths")

type Scores struct {
	MSE  float64 // mean squared error
	MAPE float64 // mean average percent error
}

func NewScores(predicted, actual []float64) (*Scores, error) {
	mse, err := MSE(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean squared error, %w", err)
	}
	mape, err := MAPE(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean average percent error, %w", err)
	}
	return &Scores{
		MSE:  mse,
		MAPE: mape,
	}, nil
}

func MSE(predicted, actual []float64) (float64, error) {
	if len(predicted) != len(actual) {
		return 0, ErrResLenMismatch
	}

	mse := 0.0
	for i := 0; i < len(actual); i++ {
		if math.IsNaN(actual[i]) || math.IsNaN(predicted[i]) {
			continue
		}
		mse += math.Pow(actual[i]-predicted[i], 2.0)
	}
	mse /= float64(len(actual))
	return mse, nil
}

func MAPE(predicted, actual []float64) (float64, error) {
	if len(predicted) != len(actual) {
		return 0, ErrResLenMismatch
	}

	mape := 0.0
	for i := 0; i < len(actual); i++ {
		if math.IsNaN(actual[i]) || math.IsNaN(predicted[i]) || actual[i] == 0 {
			continue
		}
		mape += math.Abs((actual[i] - predicted[i]) / actual[i])
	}
	mape /= float64(len(actual))
	return mape, nil
}
