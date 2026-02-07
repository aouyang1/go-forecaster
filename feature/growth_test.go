package feature

import (
	"math"
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
	endTime := time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC) // 4 days later

	testCases := []struct {
		name           string
		growth         *Growth
		epoch          []float64
		trainStartTime time.Time
		trainEndTime   time.Time
		wantLen        int
		wantNil        bool
		description    string
	}{
		{
			name:           "intercept growth",
			growth:         Intercept(),
			epoch:          []float64{1672531200, 1672617600, 1672704000}, // Unix timestamps
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        3,
			wantNil:        false,
			description:    "Intercept should return all ones",
		},
		{
			name:           "linear growth",
			growth:         Linear(),
			epoch:          []float64{1672531200, 1672617600, 1672704000, 1672790400},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        4,
			wantNil:        false,
			description:    "Linear growth should scale from 0 to 1",
		},
		{
			name:           "quadratic growth",
			growth:         Quadratic(),
			epoch:          []float64{1672531200, 1672617600, 1672704000},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        3,
			wantNil:        false,
			description:    "Quadratic growth should follow x^2 pattern",
		},
		{
			name:           "empty epoch series",
			growth:         Linear(),
			epoch:          []float64{},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        0,
			wantNil:        false,
			description:    "Empty input should return empty output",
		},
		{
			name:           "single epoch point",
			growth:         Linear(),
			epoch:          []float64{1672531200},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        1,
			wantNil:        false,
			description:    "Single epoch point should work",
		},
		{
			name:           "zero training duration",
			growth:         Linear(),
			epoch:          []float64{1672531200, 1672617600},
			trainStartTime: startTime,
			trainEndTime:   startTime, // Same time as start
			wantLen:        0,
			wantNil:        true,
			description:    "Zero duration should return nil",
		},
		{
			name:           "unknown growth type",
			growth:         NewGrowth("unknown"),
			epoch:          []float64{1672531200, 1672617600},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        0,
			wantNil:        true,
			description:    "Unknown growth type should return nil",
		},
		{
			name:           "epoch points outside training range",
			growth:         Linear(),
			epoch:          []float64{1672444800, 1672876800}, // Before and after
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        2,
			wantNil:        false,
			description:    "Epoch outside training range should work",
		},
		{
			name:           "very short training period",
			growth:         Linear(),
			epoch:          []float64{1672531200, 1672531201},
			trainStartTime: startTime,
			trainEndTime:   startTime.Add(time.Second),
			wantLen:        2,
			wantNil:        false,
			description:    "One second training period",
		},
		{
			name:           "intercept with empty epoch",
			growth:         Intercept(),
			epoch:          []float64{},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			wantLen:        0,
			wantNil:        false,
			description:    "Intercept with empty epoch should return empty slice",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.growth.Generate(tc.epoch, tc.trainStartTime, tc.trainEndTime)

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

			// Specific validations for different growth types
			if len(got) > 0 {
				switch tc.growth.Name {
				case GrowthIntercept:
					// All values should be exactly 1.0 for intercept
					for i, val := range got {
						assert.InDelta(t, 1.0, val, 1e-10, "intercept value at index %d should be 1.0", i)
					}

				case GrowthLinear:
					// Values can be outside [0, 1] range when epoch points are outside training range
					// Only validate non-negative for cases within training range
					for i, val := range got {
						assert.False(t, math.IsNaN(val), "linear growth value at index %d should not be NaN", i)
					}

				case GrowthQuadratic:
					// Values should be non-negative for squared terms
					for i, val := range got {
						assert.True(t, val >= 0.0, "quadratic growth value at index %d should be >= 0", i)
					}
				}
			}
		})
	}
}

func TestGrowthGenerateKnownValues(t *testing.T) {
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
		description    string
	}{
		{
			name:           "linear growth boundaries",
			growth:         Linear(),
			epoch:          []float64{float64(startTime.Unix()), float64(endTime.Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.0, 1.0}, // Start at 0, end at 1
			tolerance:      1e-10,
			description:    "Linear growth should go from 0 to 1",
		},
		{
			name:           "linear growth midpoint",
			growth:         Linear(),
			epoch:          []float64{float64(startTime.Add(2 * 24 * time.Hour).Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.5}, // Halfway should be 0.5
			tolerance:      1e-10,
			description:    "Linear growth midpoint should be 0.5",
		},
		{
			name:           "quadratic growth boundaries",
			growth:         Quadratic(),
			epoch:          []float64{float64(startTime.Unix()), float64(endTime.Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.0, 1.0}, // x^2 at 0 and 1
			tolerance:      1e-10,
			description:    "Quadratic growth should go from 0 to 1",
		},
		{
			name:           "quadratic growth midpoint",
			growth:         Quadratic(),
			epoch:          []float64{float64(startTime.Add(2 * 24 * time.Hour).Unix())},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{0.25}, // 0.5^2 = 0.25
			tolerance:      1e-10,
			description:    "Quadratic growth midpoint should be 0.25",
		},
		{
			name:           "intercept constant values",
			growth:         Intercept(),
			epoch:          []float64{1672531200, 1672617600, 1672704000},
			trainStartTime: startTime,
			trainEndTime:   endTime,
			expected:       []float64{1.0, 1.0, 1.0}, // All ones
			tolerance:      1e-10,
			description:    "Intercept should always return 1.0",
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

func TestGrowthFactoryFunctions(t *testing.T) {
	testCases := []struct {
		name         string
		factory      func() *Growth
		expectedName string
		description  string
	}{
		{
			name:         "Intercept factory",
			factory:      Intercept,
			expectedName: GrowthIntercept,
			description:  "Intercept factory should create intercept growth",
		},
		{
			name:         "Linear factory",
			factory:      Linear,
			expectedName: GrowthLinear,
			description:  "Linear factory should create linear growth",
		},
		{
			name:         "Quadratic factory",
			factory:      Quadratic,
			expectedName: GrowthQuadratic,
			description:  "Quadratic factory should create quadratic growth",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			growth := tc.factory()
			assert.Equal(t, tc.expectedName, growth.Name)
			assert.Equal(t, "growth_"+tc.expectedName, growth.String())
		})
	}
}

func TestGrowthConstants(t *testing.T) {
	// Test that the constants are correctly defined
	assert.Equal(t, "intercept", GrowthIntercept)
	assert.Equal(t, "linear", GrowthLinear)
	assert.Equal(t, "quadratic", GrowthQuadratic)
}
