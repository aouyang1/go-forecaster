package feature

import (
	"testing"
	"time"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChangepointString(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	expected := "chpnt_blargh_bias"
	assert.Equal(t, expected, feat.String())
}

func TestChangepointGet(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)

	testData := map[string]struct {
		label     string
		expVal    string
		expExists bool
	}{
		"unknown": {
			label: "unknown",
		},
		"capitalized": {
			label:     "NAME",
			expVal:    "blargh",
			expExists: true,
		},
		"exact match": {
			label:     "name",
			expVal:    "blargh",
			expExists: true,
		},
		"changepoint component": {
			label:     "changepoint_component",
			expVal:    "bias",
			expExists: true,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			val, exists := feat.Get(td.label)
			assert.Equal(t, td.expExists, exists, "exists")
			assert.Equal(t, td.expVal, val, "value")
		})
	}
}

func TestChangepointDecode(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	exp := map[string]string{
		"name":                  "blargh",
		"changepoint_component": "bias",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestChangepointUnmarshalJSON(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Changepoint
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}

func TestChangepointGenerate(t *testing.T) {
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	changepointTime := time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC)

	testCases := []struct {
		name        string
		changepoint *Changepoint
		times       []time.Time
		chptT       time.Time
		delta       float64
		expected    []float64
		tolerance   float64
	}{
		{
			name:        "bias component exact values",
			changepoint: NewChangepoint("test", ChangepointCompBias),
			times: []time.Time{
				baseTime,                     // before: 0
				baseTime.Add(48 * time.Hour), // at: 1
				baseTime.Add(72 * time.Hour), // after: 1
			},
			chptT:     changepointTime,
			delta:     3600.0,
			expected:  []float64{0.0, 1.0, 1.0},
			tolerance: 1e-10,
		},
		{
			name:        "slope component exact values",
			changepoint: NewChangepoint("test", ChangepointCompSlope),
			times: []time.Time{
				baseTime,                     // before: 0
				baseTime.Add(48 * time.Hour), // at: 0
				baseTime.Add(72 * time.Hour), // 1 day after: 1
				baseTime.Add(96 * time.Hour), // 2 days after: 2
			},
			chptT:     changepointTime,
			delta:     86400.0, // 1 day
			expected:  []float64{0.0, 0.0, 1.0, 2.0},
			tolerance: 1e-10,
		},
		{
			name:        "slope with fractional delta",
			changepoint: NewChangepoint("test", ChangepointCompSlope),
			times: []time.Time{
				changepointTime,                     // at: 0
				changepointTime.Add(6 * time.Hour),  // 6 hours: 0.25
				changepointTime.Add(12 * time.Hour), // 12 hours: 0.5
				changepointTime.Add(24 * time.Hour), // 24 hours: 1.0
			},
			chptT:     changepointTime,
			delta:     86400.0, // 1 day
			expected:  []float64{0.0, 0.25, 0.5, 1.0},
			tolerance: 1e-10,
		},
		{
			name:        "slope with negative delta",
			changepoint: NewChangepoint("test", ChangepointCompSlope),
			times: []time.Time{
				changepointTime,                // at: 0
				changepointTime.Add(time.Hour), // 1 hour: -1
			},
			chptT:     changepointTime,
			delta:     -3600.0, // -1 hour
			expected:  []float64{0.0, -1.0},
			tolerance: 1e-10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.changepoint.Generate(tc.times, tc.chptT, tc.delta)

			assert.NotNil(t, got)
			assert.Len(t, got, len(tc.expected))

			for i, expected := range tc.expected {
				assert.InDelta(t, expected, got[i], tc.tolerance,
					"value at index %d", i)
			}
		})
	}
}
