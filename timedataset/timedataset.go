package timedataset

import (
	"errors"
	"fmt"
	"math"
	"time"
)

var (
	ErrNoTrainingData     = errors.New("no training data")
	ErrNonMontonic        = errors.New("time feature is not monotonic")
	ErrDatasetLenMismatch = errors.New("time feature has a different length than observations")
	ErrCannotInferFreq    = errors.New("cannot infer frequency from time data")
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

func (td *TimeDataset) Len() int {
	if td == nil {
		return 0
	}
	return len(td.T)
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

// DropNan drops of records where the Y variable is NaN. This assumes the data is
// in time sorted order already. This creates a new TimeDataset
func (td *TimeDataset) DropNan() *TimeDataset {
	if td == nil {
		return nil
	}

	tdCopy := td.Copy()

	// drop out nans
	var ptr int
	for i := 0; i < len(tdCopy.T); i++ {
		if math.IsNaN(tdCopy.Y[i]) {
			continue
		}
		tdCopy.T[ptr] = tdCopy.T[i]
		tdCopy.Y[ptr] = tdCopy.Y[i]
		ptr++
	}
	tdCopy.T = tdCopy.T[:ptr]
	tdCopy.Y = tdCopy.Y[:ptr]
	return tdCopy
}
