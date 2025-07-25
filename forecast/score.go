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
	if len(predicted) != len(actual) {
		return nil, fmt.Errorf("expected %d, but got %d, %w", len(actual), len(predicted), ErrResLenMismatch)
	}

	mseScore, err := mse(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean squared error, %w", err)
	}
	mapeScore, err := mape(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute mean average percent error, %w", err)
	}
	rsScore, err := rsquared(predicted, actual)
	if err != nil {
		return nil, fmt.Errorf("unable to compute r-squared, %w", err)
	}

	return &Scores{
		MSE:  mseScore,
		MAPE: mapeScore,
		R2:   rsScore,
	}, nil
}

// MSE computes the mean squared error. This is the same as sum((y-yhat)^2).
// A score of 0 means a perfect match with no errors.
func mse(predicted, actual []float64) (float64, error) {
	mse := 0.0
	for i := range len(actual) {
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
func mape(predicted, actual []float64) (float64, error) {
	mape := 0.0
	for i := range len(actual) {
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
func rsquared(predicted, actual []float64) (float64, error) {
	predictCopy := make([]float64, 0, len(predicted))
	actualCopy := make([]float64, 0, len(actual))
	for i := range len(predicted) {
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
