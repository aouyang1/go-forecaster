package timedataset

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"gonum.org/v1/gonum/floats"
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

func GenerateT(n int, interval time.Duration, nowFunc func() time.Time) []time.Time {
	t := make([]time.Time, 0, n)
	ct := time.Unix(nowFunc().Unix()/60*60, 0).Add(-time.Duration(n) * interval)
	for i := 0; i < n; i++ {
		t = append(t, ct.Add(interval*time.Duration(i)))
	}
	return t
}

type Series []float64

func (s Series) Add(src Series) Series {
	floats.Add(s, src)
	return s
}

func (s Series) SetConst(t []time.Time, val float64, start, end time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		if (t[i].After(start) || t[i].Equal(start)) && t[i].Before(end) {
			s[i] = val
		}
	}
	return s
}

func (s Series) MaskWithWeekend(t []time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		switch t[i].Weekday() {
		case time.Saturday, time.Sunday:
			continue
		default:
			s[i] = 0.0
		}
	}
	return s
}

func (s Series) MaskWithTimeRange(start, end time.Time, t []time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		if t[i].Before(start) || t[i].After(end) {
			s[i] = 0.0
		}
	}
	return s
}

func GenerateConstY(n int, val float64) Series {
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		y = append(y, val)
	}
	return Series(y)
}

func GenerateWaveY(t []time.Time, amp, periodSec, order, timeOffset float64) Series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		val := amp * math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset))
		y = append(y, val)
	}
	return Series(y)
}

func GenerateNoise(t []time.Time, noiseScale, amp, periodSec, order, timeOffset float64) Series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		scale := (noiseScale + amp*math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset)))
		y = append(y, rand.NormFloat64()*scale)
	}
	return Series(y)
}

func GenerateChange(t []time.Time, chpt time.Time, bias, slope float64) Series {
	n := len(t)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		if t[i].After(chpt) || t[i].Equal(chpt) {
			jump := bias + slope*t[i].Sub(chpt).Minutes()
			y[i] = jump
		}
	}
	return Series(y)
}
