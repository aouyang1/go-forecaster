package feature

import (
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
