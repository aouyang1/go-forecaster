package feature

import (
	"testing"
	"time"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGrowthString(t *testing.T) {
	feat := NewGrowth("linear")
	expected := "growth_linear"
	assert.Equal(t, expected, feat.String())
}

func TestGrowthGet(t *testing.T) {
	feat := NewGrowth("linear")

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
			expVal:    "linear",
			expExists: true,
		},
		"exact match": {
			label:     "name",
			expVal:    "linear",
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

func TestGrowthDecode(t *testing.T) {
	feat := NewGrowth("quadratic")
	exp := map[string]string{
		"name": "quadratic",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestGrowthUnmarshalJSON(t *testing.T) {
	feat := NewGrowth("intercept")
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Growth
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}

func TestGrowthGenerate(t *testing.T) {
	startTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)
	endTime := time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC) // 4 days = 345600 seconds

	testCases := []struct {
		name           string
		growth         *Growth
		epoch          []float64
		trainStartTime time.Time
		trainEndTime   time.Time
		expected       []float64
		tolerance      float64
	}{
		{
			name:           "linear growth boundaries",
			growth:         Linear(),
			epoch:          []float64{float64(startTime.Unix()), float64(endTime.Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.0, 1.0}, // Start at 0, end at 1
			tolerance:      1e-10,
		},
		{
			name:           "linear growth midpoint",
			growth:         Linear(),
			epoch:          []float64{float64(startTime.Add(2 * 24 * time.Hour).Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.5}, // Halfway should be 0.5
			tolerance:      1e-10,
		},
		{
			name:           "quadratic growth boundaries",
			growth:         Quadratic(),
			epoch:          []float64{float64(startTime.Unix()), float64(endTime.Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.0, 1.0}, // x^2 at 0 and 1
			tolerance:      1e-10,
		},
		{
			name:           "quadratic growth midpoint",
			growth:         Quadratic(),
			epoch:          []float64{float64(startTime.Add(2 * 24 * time.Hour).Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.25}, // 0.5^2 = 0.25
			tolerance:      1e-10,
		},
		{
			name:           "intercept constant values",
			growth:         Intercept(),
			epoch:          []float64{1672531200, 1672617600, 1672704000},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{1.0, 1.0, 1.0}, // All ones
			tolerance:      1e-10,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.growth.Generate(tc.epoch, tc.trainStartTime, tc.trainEndTime)

			assert.Len(t, got, len(tc.expected))

			for i, expected := range tc.expected {
				assert.InDelta(t, expected, got[i], tc.tolerance,
					"value at index %d", i)
			}
		})
	}
}
