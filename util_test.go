package forecaster

import (
	"math"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/stretchr/testify/assert"
)

func assertFloatSliceEqualWithNaN(t *testing.T, expected, actual []float64) {
	t.Helper()
	if len(expected) != len(actual) {
		assert.Failf(t, "length mismatch", "expected len=%d, got len=%d", len(expected), len(actual))
		return
	}
	for i := range expected {
		e, a := expected[i], actual[i]
		if math.IsNaN(e) && math.IsNaN(a) {
			continue
		}
		assert.Equalf(t, e, a, "index %d mismatch", i)
	}
}

func TestRemoveOutlierEvents(t *testing.T) {
	start := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

	type input struct {
		t      []time.Time
		y      []float64
		events []options.Event
	}

	testData := map[string]struct {
		input     input
		expectedY []float64
	}{
		"no events": {
			input: input{
				t: []time.Time{
					start,
					start.Add(time.Hour),
					start.Add(2 * time.Hour),
				},
				y: []float64{1, 2, 3},
			},
			expectedY: []float64{1, 2, 3},
		},
		"events outside range": {
			input: input{
				t: []time.Time{
					start,
					start.Add(time.Hour),
					start.Add(2 * time.Hour),
				},
				y: []float64{1, 2, 3},
				events: []options.Event{
					{
						Name:  "before",
						Start: start.Add(-4 * time.Hour),
						End:   start.Add(-2 * time.Hour),
					},
					{
						Name:  "after",
						Start: start.Add(4 * time.Hour),
						End:   start.Add(6 * time.Hour),
					},
				},
			},
			expectedY: []float64{1, 2, 3},
		},
		"single event inclusive of bounds": {
			input: input{
				t: []time.Time{
					start,
					start.Add(time.Hour),
					start.Add(2 * time.Hour),
					start.Add(3 * time.Hour),
					start.Add(4 * time.Hour),
				},
				y: []float64{1, 2, 3, 4, 5},
				events: []options.Event{
					{
						Name:  "event1",
						Start: start.Add(time.Hour),
						End:   start.Add(3 * time.Hour),
					},
				},
			},
			// indices 1,2,3 should be removed (inclusive of start & end)
			expectedY: []float64{
				1,
				math.NaN(),
				math.NaN(),
				math.NaN(),
				5,
			},
		},
		"overlapping events": {
			input: input{
				t: []time.Time{
					start,
					start.Add(time.Hour),
					start.Add(2 * time.Hour),
					start.Add(3 * time.Hour),
					start.Add(4 * time.Hour),
				},
				y: []float64{10, 20, 30, 40, 50},
				events: []options.Event{
					{
						Name:  "event1",
						Start: start.Add(time.Hour),
						End:   start.Add(3 * time.Hour),
					},
					{
						Name:  "event2",
						Start: start.Add(2 * time.Hour),
						End:   start.Add(4 * time.Hour),
					},
				},
			},
			// indices 1-4 should all be removed
			expectedY: []float64{
				10,
				math.NaN(),
				math.NaN(),
				math.NaN(),
				math.NaN(),
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			// copy input so each test is isolated
			y := make([]float64, len(td.input.y))
			copy(y, td.input.y)

			removeOutlierEvents(td.input.t, y, td.input.events)

			assertFloatSliceEqualWithNaN(t, td.expectedY, y)
		})
	}
}

func TestAutoRemoveOutliers(t *testing.T) {
	start := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)

	type input struct {
		y        []float64
		residual []float64
		opts     *OutlierOptions
	}

	testData := map[string]struct {
		input         input
		expectedY     []float64
		expectedCount int
	}{
		"nil options": {
			input: input{
				y:        []float64{1, 2, 3},
				residual: []float64{0.1, 0.2, 0.3},
				opts:     nil,
			},
			expectedY:     []float64{1, 2, 3},
			expectedCount: 0,
		},
		"zero passes": {
			input: input{
				y:        []float64{1, 2, 3},
				residual: []float64{0.1, 0.2, 0.3},
				opts: &OutlierOptions{
					NumPasses:       0,
					UpperPercentile: 0.8,
					LowerPercentile: 0.2,
					TukeyFactor:     0.0,
				},
			},
			expectedY:     []float64{1, 2, 3},
			expectedCount: 0,
		},
		"no outliers detected": {
			input: input{
				y:        []float64{1, 2, 3, 4, 5},
				residual: []float64{0, 0.1, 0.2, 0.15, 0.05},
				opts: &OutlierOptions{
					NumPasses:       1,
					UpperPercentile: 0.8,
					LowerPercentile: 0.2,
					TukeyFactor:     1.5,
				},
			},
			expectedY:     []float64{1, 2, 3, 4, 5},
			expectedCount: 0,
		},
		"single outlier": {
			input: input{
				y:        []float64{1, 2, 3, 4, 5},
				residual: []float64{0, 0, 0, 10, 0},
				opts: &OutlierOptions{
					NumPasses:       1,
					UpperPercentile: 0.6,
					LowerPercentile: 0.2,
					TukeyFactor:     0.0,
				},
			},
			// index 3 should be set to NaN
			expectedY: []float64{
				1,
				2,
				3,
				math.NaN(),
				5,
			},
			expectedCount: 1,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			y := make([]float64, len(td.input.y))
			copy(y, td.input.y)

			// build a time slice of matching length; actual values are irrelevant
			times := make([]time.Time, len(y))
			for i := range times {
				times[i] = start.Add(time.Duration(i) * time.Minute)
			}

			count := autoRemoveOutliers(times, y, td.input.residual, td.input.opts)

			assert.Equal(t, td.expectedCount, count)
			assertFloatSliceEqualWithNaN(t, td.expectedY, y)
		})
	}
}
