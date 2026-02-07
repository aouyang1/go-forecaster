package feature

import (
	"math"
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
		wantLen     int
		wantNil     bool
		description string
	}{
		{
			name:        "bias component basic case",
			changepoint: NewChangepoint("test", ChangepointCompBias),
			times: []time.Time{
				baseTime,                     // before changepoint
				baseTime.Add(24 * time.Hour), // before changepoint
				baseTime.Add(48 * time.Hour), // at changepoint
				baseTime.Add(72 * time.Hour), // after changepoint
				baseTime.Add(96 * time.Hour), // after changepoint
			},
			chptT:       changepointTime,
			delta:       3600.0, // 1 hour (not used for bias)
			wantLen:     5,
			wantNil:     false,
			description: "Bias component should be 0 before changepoint, 1 after/at changepoint",
		},
		{
			name:        "slope component basic case",
			changepoint: NewChangepoint("test", ChangepointCompSlope),
			times: []time.Time{
				baseTime,                     // before changepoint
				baseTime.Add(24 * time.Hour), // before changepoint
				baseTime.Add(48 * time.Hour), // at changepoint
				baseTime.Add(72 * time.Hour), // 1 day after changepoint
				baseTime.Add(96 * time.Hour), // 2 days after changepoint
			},
			chptT:       changepointTime,
			delta:       86400.0, // 1 day
			wantLen:     5,
			wantNil:     false,
			description: "Slope component should be 0 before changepoint, increase linearly after",
		},
		{
			name:        "empty time series",
			changepoint: NewChangepoint("empty", ChangepointCompBias),
			times:       []time.Time{},
			chptT:       changepointTime,
			delta:       3600.0,
			wantLen:     0,
			wantNil:     false,
			description: "Empty time series should return empty result",
		},
		{
			name:        "single time point before changepoint",
			changepoint: NewChangepoint("single", ChangepointCompBias),
			times: []time.Time{
				baseTime,
			},
			chptT:       changepointTime.Add(24 * time.Hour), // changepoint after single point
			delta:       3600.0,
			wantLen:     1,
			wantNil:     false,
			description: "Single point before changepoint should be 0 for bias",
		},
		{
			name:        "single time point at changepoint",
			changepoint: NewChangepoint("single", ChangepointCompSlope),
			times: []time.Time{
				changepointTime,
			},
			chptT:       changepointTime,
			delta:       3600.0,
			wantLen:     1,
			wantNil:     false,
			description: "Single point at changepoint should be 0 for slope",
		},
		{
			name:        "all points before changepoint",
			changepoint: NewChangepoint("before", ChangepointCompBias),
			times: []time.Time{
				baseTime,
				baseTime.Add(12 * time.Hour),
				baseTime.Add(24 * time.Hour),
			},
			chptT:       baseTime.Add(48 * time.Hour), // changepoint after all points
			delta:       3600.0,
			wantLen:     3,
			wantNil:     false,
			description: "All points before changepoint should be 0",
		},
		{
			name:        "all points after changepoint",
			changepoint: NewChangepoint("after", ChangepointCompBias),
			times: []time.Time{
				changepointTime.Add(24 * time.Hour),
				changepointTime.Add(48 * time.Hour),
				changepointTime.Add(72 * time.Hour),
			},
			chptT:       changepointTime,
			delta:       3600.0,
			wantLen:     3,
			wantNil:     false,
			description: "All points after changepoint should be 1 for bias",
		},
		{
			name:        "slope with zero delta should return nil",
			changepoint: NewChangepoint("zero_delta", ChangepointCompSlope),
			times: []time.Time{
				baseTime,
				baseTime.Add(48 * time.Hour),
				baseTime.Add(96 * time.Hour),
			},
			chptT:       changepointTime,
			delta:       0.0,
			wantLen:     0,
			wantNil:     true,
			description: "Slope component with zero delta should return nil",
		},
		{
			name:        "bias with zero delta should work",
			changepoint: NewChangepoint("zero_delta_bias", ChangepointCompBias),
			times: []time.Time{
				baseTime,
				baseTime.Add(48 * time.Hour),
				baseTime.Add(96 * time.Hour),
			},
			chptT:       changepointTime,
			delta:       0.0,
			wantLen:     3,
			wantNil:     false,
			description: "Bias component with zero delta should work normally",
		},
		{
			name:        "slope with small delta",
			changepoint: NewChangepoint("small_delta", ChangepointCompSlope),
			times: []time.Time{
				baseTime.Add(24 * time.Hour), // 1 day before
				changepointTime,              // at changepoint
				baseTime.Add(72 * time.Hour), // 1 day after
			},
			chptT:       changepointTime,
			delta:       3600.0, // 1 hour
			wantLen:     3,
			wantNil:     false,
			description: "Slope with small delta should produce large values",
		},
		{
			name:        "irregular time intervals",
			changepoint: NewChangepoint("irregular", ChangepointCompBias),
			times: []time.Time{
				baseTime,
				baseTime.Add(12 * time.Hour), // 12 hours
				baseTime.Add(48 * time.Hour), // 36 hours
				baseTime.Add(61 * time.Hour), // 13 hours
			},
			chptT:       baseTime.Add(36 * time.Hour), // irregular changepoint time
			delta:       3600.0,
			wantLen:     4,
			wantNil:     false,
			description: "Irregular time intervals should work correctly",
		},
		{
			name:        "times with nanosecond precision",
			changepoint: NewChangepoint("nanos", ChangepointCompSlope),
			times: []time.Time{
				changepointTime,
				changepointTime.Add(time.Hour),
				changepointTime.Add(time.Hour + time.Millisecond),
			},
			chptT:       changepointTime,
			delta:       3600.0,
			wantLen:     3,
			wantNil:     false,
			description: "Nanosecond precision should be handled correctly",
		},
		{
			name:        "negative delta",
			changepoint: NewChangepoint("neg_delta", ChangepointCompSlope),
			times: []time.Time{
				baseTime.Add(48 * time.Hour), // at changepoint
				baseTime.Add(72 * time.Hour), // after
			},
			chptT:       changepointTime,
			delta:       -3600.0,
			wantLen:     2,
			wantNil:     false,
			description: "Negative delta should work (produces negative slope)",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.changepoint.Generate(tc.times, tc.chptT, tc.delta)

			if tc.wantNil {
				assert.Nil(t, got)
				return
			}

			assert.NotNil(t, got)
			assert.Len(t, got, tc.wantLen)

			// Validate output values
			for i, val := range got {
				assert.False(t, math.IsNaN(val), "value at index %d should not be NaN", i)
				assert.False(t, math.IsInf(val, 0), "value at index %d should not be infinite", i)
			}

			// Component-specific validations
			if tc.changepoint.ChangepointComp == ChangepointCompBias && len(got) > 0 {
				// Bias component should have values 0 or 1
				for i, val := range got {
					assert.True(t, val == 0.0 || val == 1.0,
						"bias value at index %d should be 0 or 1, got %f", i, val)
				}
			}

			if tc.changepoint.ChangepointComp == ChangepointCompSlope && len(got) > 0 && tc.delta != 0 {
				// Before changepoint should be 0 for slope (but we can't check after without more complex logic)
				for i, tPoint := range tc.times {
					if tPoint.Before(tc.chptT) {
						assert.InDelta(t, 0.0, got[i], 1e-10,
							"slope value before changepoint at index %d should be 0", i)
					}
				}
			}
		})
	}
}

func TestChangepointGenerateKnownValues(t *testing.T) {
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
		description string
	}{
		{
			name:        "bias component exact values",
			changepoint: NewChangepoint("test", ChangepointCompBias),
			times: []time.Time{
				baseTime,                     // before: 0
				baseTime.Add(48 * time.Hour), // at: 1
				baseTime.Add(72 * time.Hour), // after: 1
			},
			chptT:       changepointTime,
			delta:       3600.0,
			expected:    []float64{0.0, 1.0, 1.0},
			tolerance:   1e-10,
			description: "Bias component should have exact 0/1 values",
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
			chptT:       changepointTime,
			delta:       86400.0, // 1 day
			expected:    []float64{0.0, 0.0, 1.0, 2.0},
			tolerance:   1e-10,
			description: "Slope component should increase by 1 per day",
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
			chptT:       changepointTime,
			delta:       86400.0, // 1 day
			expected:    []float64{0.0, 0.25, 0.5, 1.0},
			tolerance:   1e-10,
			description: "Slope component with fractional increments",
		},
		{
			name:        "slope with negative delta",
			changepoint: NewChangepoint("test", ChangepointCompSlope),
			times: []time.Time{
				changepointTime,                // at: 0
				changepointTime.Add(time.Hour), // 1 hour: -1
			},
			chptT:       changepointTime,
			delta:       -3600.0, // -1 hour
			expected:    []float64{0.0, -1.0},
			tolerance:   1e-10,
			description: "Slope with negative delta should produce negative values",
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

func TestChangepointConstants(t *testing.T) {
	// Test that the constants are correctly defined
	assert.Equal(t, "bias", ChangepointCompBias)
	assert.Equal(t, "slope", ChangepointCompSlope)
}

func TestChangepointFactoryFunctions(t *testing.T) {
	// Test the factory function behavior
	biasChangepoint := NewChangepoint("test", ChangepointCompBias)
	assert.Equal(t, string(ChangepointCompBias), string(biasChangepoint.ChangepointComp))
	assert.Equal(t, "chpnt_test_bias", biasChangepoint.String())

	slopeChangepoint := NewChangepoint("test", ChangepointCompSlope)
	assert.Equal(t, string(ChangepointCompSlope), string(slopeChangepoint.ChangepointComp))
	assert.Equal(t, "chpnt_test_slope", slopeChangepoint.String())
}
