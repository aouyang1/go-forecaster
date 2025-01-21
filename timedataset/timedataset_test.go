package timedataset

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewUnivariateDataset(t *testing.T) {
	testData := map[string]struct {
		t        []time.Time
		y        []float64
		expected *TimeDataset
		err      error
	}{
		"no training data": {
			err: ErrNoTrainingData,
		},
		"length mismatch": {
			y:   []float64{1},
			err: ErrDatasetLenMismatch,
		},
		"non increasing time": {
			t: []time.Time{
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
			},
			y:   []float64{1, 2},
			err: ErrNonMontonic,
		},
		"valid": {
			t: []time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			y: []float64{1, 2},
			expected: &TimeDataset{
				T: []time.Time{
					time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				},
				Y: []float64{1, 2},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			ds, err := NewUnivariateDataset(td.t, td.y)
			if td.err != nil {
				assert.ErrorAs(t, err, &td.err)
				return
			}
			assert.Equal(t, td.expected, ds)
		})
	}
}

func TestCopy(t *testing.T) {
	tSeries := []time.Time{
		time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
		time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
	}

	y := []float64{0, 1}
	ds, err := NewUnivariateDataset(tSeries, y)
	require.Nil(t, err)

	nextDs := ds.Copy()
	require.Equal(t, ds, nextDs)

	ds.T = []time.Time{
		time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
		time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
	}
	require.NotEqual(t, nextDs, ds)
}

func TestDropNan(t *testing.T) {
	testData := map[string]struct {
		tdset    *TimeDataset
		expected *TimeDataset
	}{
		"nil input for nan drop": {tdset: nil, expected: nil},
		"no data to drop": {
			tdset: &TimeDataset{},
			expected: &TimeDataset{
				T: []time.Time{},
				Y: []float64{},
			},
		},
		"no NaNs": {
			tdset: &TimeDataset{
				T: []time.Time{
					time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
				},
				Y: []float64{1, 2, 3, 4},
			},
			expected: &TimeDataset{
				T: []time.Time{
					time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
				},
				Y: []float64{1, 2, 3, 4},
			},
		},
		"data with NaNs": {
			tdset: &TimeDataset{
				T: []time.Time{
					time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 6, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 7, 0, 0, 0, 0, time.UTC),
				},
				Y: []float64{math.NaN(), 2, 3, math.NaN(), 5, 6, math.NaN()},
			},
			expected: &TimeDataset{
				T: []time.Time{
					time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
					time.Date(1970, 1, 6, 0, 0, 0, 0, time.UTC),
				},
				Y: []float64{2, 3, 5, 6},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.tdset.DropNan()
			assert.Equal(t, td.expected, res)
		})
	}
}
