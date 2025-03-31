package forecast

import (
	"errors"
	"fmt"
	"math"

	"gonum.org/v1/gonum/stat"
)

var ErrResLenMismatch = errors.New("predicted and actual have different lengths")

// Scores tracks the fit scores
type Scores struct {
	MSE  float64 `json:"mean_squared_error"`
	MAPE float64 `json:"mean_average_percent_error"`
	R2   float64 `json:"r_squared"`
}

// NewScores calculates the fit scores given the predicted and actual input slice values
func NewScores(predicted, actual []float64) (*Scores, error) {
	mse, err := MSE(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean squared error, %w", err)
	}
	mape, err := MAPE(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean average percent error, %w", err)
	}
	rs, err := RSquared(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute r-squared, %w", err)
	}

	return &Scores{
		MSE:  mse,
		MAPE: mape,
		R2:   rs,
	}, nil
}

// MSE computes the mean squared error. This is the same as sum((y-yhat)^2).
// A score of 0 means a perfect match with no errors.
func MSE(predicted, actual []float64) (float64, error) {
	if len(predicted) != len(actual) {
		return 0, fmt.Errorf("expected %d, but got %d, %w", len(actual), len(predicted), ErrResLenMismatch)
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

// MAPE calculates the mean average percent error. This is the same as sum(abs((y-yhat)/y)).
// A score of 0 means a perfect match with no errors.
func MAPE(predicted, actual []float64) (float64, error) {
	if len(predicted) != len(actual) {
		return 0, fmt.Errorf("expected %d, but got %d, %w", len(actual), len(predicted), ErrResLenMismatch)
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

// RSquared computes the r squared value between the predicted and actual where 1.0 means perfect
// fit and 0 represents no relationship
func RSquared(predicted, actual []float64) (float64, error) {
	if len(predicted) != len(actual) {
		return 0, fmt.Errorf("expected %d, but got %d, %w", len(actual), len(predicted), ErrResLenMismatch)
	}

	predictCopy := make([]float64, 0, len(predicted))
	actualCopy := make([]float64, 0, len(actual))
	for i := 0; i < len(predicted); i++ {
		if math.IsNaN(actual[i]) || math.IsNaN(predicted[i]) {
			continue
		}
		predictCopy = append(predictCopy, predicted[i])
		actualCopy = append(actualCopy, actual[i])
	}
	r2 := stat.RSquaredFrom(predictCopy, actualCopy, nil)
	if math.IsNaN(r2) {
		return 1.0, nil
	}
	return r2, nil
}
