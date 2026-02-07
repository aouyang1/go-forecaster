package feature

import (
	"math"
	"testing"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSeasonalityString(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	expected := "seas_hod_2_cos"
	assert.Equal(t, expected, feat.String())
}

func TestSeasonalityGet(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)

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
			expVal:    "hod",
			expExists: true,
		},
		"exact match": {
			label:     "name",
			expVal:    "hod",
			expExists: true,
		},
		"fourier component": {
			label:     "fourier_component",
			expVal:    "cos",
			expExists: true,
		},
		"order": {
			label:     "order",
			expVal:    "2",
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

func TestSeasonalityDecode(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	exp := map[string]string{
		"name":              "hod",
		"fourier_component": "cos",
		"order":             "2",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestSeasonalityUnmarshalJSON(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Seasonality
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}

func TestSeasonalityGenerate(t *testing.T) {
	testCases := []struct {
		name        string
		seasonality *Seasonality
		timePoints  []float64
		order       int
		period      float64
		wantLen     int
	}{
		{
			name:        "sin component with basic values",
			seasonality: NewSeasonality("daily", FourierCompSin, 1),
			timePoints:  []float64{0, 0.25, 0.5, 0.75, 1.0},
			order:       1,
			period:      1.0,
			wantLen:     5,
		},
		{
			name:        "cos component with basic values",
			seasonality: NewSeasonality("daily", FourierCompCos, 1),
			timePoints:  []float64{0, 0.25, 0.5, 0.75, 1.0},
			order:       1,
			period:      1.0,
			wantLen:     5,
		},
		{
			name:        "sin component with higher order",
			seasonality: NewSeasonality("daily", FourierCompSin, 2),
			timePoints:  []float64{0, 0.125, 0.25, 0.375, 0.5},
			order:       2,
			period:      1.0,
			wantLen:     5,
		},
		{
			name:        "cos component with higher order",
			seasonality: NewSeasonality("daily", FourierCompCos, 3),
			timePoints:  []float64{0, 0.08333, 0.16667, 0.25, 0.33333},
			order:       3,
			period:      1.0,
			wantLen:     5,
		},
		{
			name:        "different period",
			seasonality: NewSeasonality("weekly", FourierCompSin, 1),
			timePoints:  []float64{0, 1, 2, 3, 4, 5, 6, 7},
			order:       1,
			period:      7.0,
			wantLen:     8,
		},
		{
			name:        "empty time series",
			seasonality: NewSeasonality("test", FourierCompSin, 1),
			timePoints:  []float64{},
			order:       1,
			period:      1.0,
			wantLen:     0,
		},
		{
			name:        "single time point",
			seasonality: NewSeasonality("test", FourierCompCos, 1),
			timePoints:  []float64{0.5},
			order:       1,
			period:      1.0,
			wantLen:     1,
		},
		{
			name:        "negative time points",
			seasonality: NewSeasonality("test", FourierCompSin, 1),
			timePoints:  []float64{-2, -1, 0, 1, 2},
			order:       1,
			period:      2.0,
			wantLen:     5,
		},
		{
			name:        "zero period (edge case)",
			seasonality: NewSeasonality("test", FourierCompCos, 1),
			timePoints:  []float64{0, 1, 2},
			order:       1,
			period:      0.0001, // Very small period to avoid division by zero
			wantLen:     3,
		},
		{
			name:        "large order",
			seasonality: NewSeasonality("test", FourierCompSin, 10),
			timePoints:  []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
			order:       10,
			period:      1.0,
			wantLen:     10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.seasonality.Generate(tc.timePoints, tc.order, tc.period)

			assert.Len(t, got, tc.wantLen)

			// Validate output values are within valid range [-1, 1]
			for i, val := range got {
				assert.True(t, val >= -1.0 && val <= 1.0,
					"value at index %d should be between -1 and 1, got %f", i, val)
				assert.False(t, math.IsNaN(val),
					"value at index %d should not be NaN", i)
				assert.False(t, math.IsInf(val, 0),
					"value at index %d should not be infinite", i)
			}

			// Additional validations for specific cases
			if len(tc.timePoints) > 0 && tc.period > 0 && tc.order > 0 {
				// Test periodicity properties for well-formed cases
				if tc.timePoints[0] == 0 && tc.seasonality.FourierComp == FourierCompCos {
					// Cosine at time 0 should be cos(0) = 1 for any order
					assert.InDelta(t, 1.0, got[0], 1e-10,
						"cosine component at time 0 should be 1")
				}

				if tc.timePoints[0] == 0 && tc.seasonality.FourierComp == FourierCompSin {
					// Sine at time 0 should be sin(0) = 0 for any order
					assert.InDelta(t, 0.0, got[0], 1e-10,
						"sine component at time 0 should be 0")
				}
			}
		})
	}
}

func TestSeasonalityGenerateKnownValues(t *testing.T) {
	// Test specific known values to ensure mathematical correctness
	testCases := []struct {
		name        string
		seasonality *Seasonality
		timePoints  []float64
		order       int
		period      float64
		expected    []float64
		tolerance   float64
	}{
		{
			name:        "cos at key points",
			seasonality: NewSeasonality("test", FourierCompCos, 1),
			timePoints:  []float64{0, 0.25, 0.5, 0.75, 1.0},
			order:       1,
			period:      1.0,
			expected:    []float64{1.0, 0.0, -1.0, 0.0, 1.0}, // cos(0), cos(π/2), cos(π), cos(3π/2), cos(2π)
			tolerance:   1e-10,
		},
		{
			name:        "sin at key points",
			seasonality: NewSeasonality("test", FourierCompSin, 1),
			timePoints:  []float64{0, 0.25, 0.5, 0.75, 1.0},
			order:       1,
			period:      1.0,
			expected:    []float64{0.0, 1.0, 0.0, -1.0, 0.0}, // sin(0), sin(π/2), sin(π), sin(3π/2), sin(2π)
			tolerance:   1e-10,
		},
		{
			name:        "second order cos",
			seasonality: NewSeasonality("test", FourierCompCos, 2),
			timePoints:  []float64{0, 0.25, 0.5},
			order:       2,
			period:      1.0,
			expected:    []float64{1.0, -1.0, 1.0}, // cos(0), cos(π), cos(2π)
			tolerance:   1e-10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.seasonality.Generate(tc.timePoints, tc.order, tc.period)

			assert.Len(t, got, len(tc.expected))

			for i, expected := range tc.expected {
				assert.InDelta(t, expected, got[i], tc.tolerance,
					"value at index %d", i)
			}
		})
	}
}
