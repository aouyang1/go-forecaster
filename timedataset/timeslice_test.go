package timedataset

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStartTime(t *testing.T) {
	testData := map[string]struct {
		tSlice   TimeSlice
		expected time.Time
	}{
		"nil input for start time": {
			tSlice:   nil,
			expected: time.Time{},
		},
		"valid start time": {
			tSlice: TimeSlice([]time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
			}),
			expected: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.tSlice.StartTime()
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestEndTime(t *testing.T) {
	testData := map[string]struct {
		tSlice   TimeSlice
		expected time.Time
	}{
		"nil input for end time": {
			tSlice:   nil,
			expected: time.Time{},
		},
		"valid end time": {
			tSlice: TimeSlice([]time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
			}),
			expected: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.tSlice.EndTime()
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestEstimateFreq(t *testing.T) {
	testData := map[string]struct {
		tSlice   TimeSlice
		expected time.Duration
		err      error
	}{
		"estimate with nil timedataset": {
			tSlice: nil,
			err:    ErrCannotInferFreq,
		},
		"consistent frequencies": {
			tSlice: TimeSlice([]time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
			}),
			expected: 24 * time.Hour,
		},
		"multiple frequencies": {
			tSlice: TimeSlice([]time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 1, 0, 0, 0, time.UTC),
			}),
			expected: 24 * time.Hour,
		},
		"multiple frequencies with same counts": {
			tSlice: TimeSlice([]time.Time{
				time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 1, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 3, 2, 0, 0, 0, time.UTC),
			}),
			expected: time.Hour,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			freq, err := td.tSlice.EstimateFreq()
			if td.err != nil {
				assert.EqualError(t, err, td.err.Error())
				return
			}
			require.NoError(t, err)
			assert.Equal(t, td.expected, freq)
		})
	}
}
