package timedataset

import (
	"errors"
	"fmt"
	"time"
)

var ErrDatasetLenMismatch = errors.New("time feature has a different length than observations")

type TimeDataset struct {
	T []time.Time
	Y []float64
}

func NewUnivariateDataset(t []time.Time, y []float64) (*TimeDataset, error) {
	if len(t) != len(y) {
		return nil, fmt.Errorf(
			"time feature has length of %d, but values has a length of %d, %w",
			len(t), len(y), ErrDatasetLenMismatch,
		)
	}

	tSeries := make([]time.Time, len(t))
	ySeries := make([]float64, len(t))
	copy(tSeries, t)
	copy(ySeries, y)
	td := &TimeDataset{
		T: tSeries,
		Y: ySeries,
	}

	return td, nil
}
