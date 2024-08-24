package main

import (
	"errors"
	"fmt"
	"time"
)

var ErrDatasetLenMismatch = errors.New("time feature has a different length than observations")

type TimeDataset struct {
	t []time.Time
	y []float64
}

func NewUnivariateDataset(t []time.Time, y []float64) (*TimeDataset, error) {
	if len(t) != len(y) {
		return nil, fmt.Errorf(
			"time feature has length of %d, but values has a length of %d, %w",
			len(t), len(y), ErrDatasetLenMismatch,
		)
	}

	td := &TimeDataset{
		t: t,
		y: y,
	}

	return td, nil
}
