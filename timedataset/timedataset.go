package timedataset

import (
	"errors"
	"fmt"
	"time"
)

var (
	ErrNoTrainingData     = errors.New("no training data")
	ErrNonMontonic        = errors.New("time feature is not monotonic")
	ErrDatasetLenMismatch = errors.New("time feature has a different length than observations")
)

// TimeDataset represents a time series storing a slice of time points and values.
// Both must be of the same length.
type TimeDataset struct {
	T []time.Time
	Y []float64
}

// NewUnivariateDataset returns an instance of a TimeDataset given a time and value slice.
func NewUnivariateDataset(t []time.Time, y []float64) (*TimeDataset, error) {
	if len(y) == 0 {
		return nil, ErrNoTrainingData
	}
	if len(t) != len(y) {
		return nil, fmt.Errorf(
			"time feature has length of %d, but values has a length of %d, %w",
			len(t), len(y), ErrDatasetLenMismatch,
		)
	}

	var lastT time.Time
	for i := 0; i < len(t); i++ {
		currT := t[i]
		if currT.Before(lastT) || currT.Equal(lastT) {
			return nil, fmt.Errorf("non-monotonic at %d, %w", i, ErrNonMontonic)
		}
		lastT = currT
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

func (td *TimeDataset) Copy() *TimeDataset {
	tSeries := make([]time.Time, len(td.T))
	ySeries := make([]float64, len(td.T))
	copy(tSeries, td.T)
	copy(ySeries, td.Y)
	return &TimeDataset{
		T: tSeries,
		Y: ySeries,
	}
}
