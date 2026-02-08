package feature

import (
	"math"
	"testing"
	"time"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimeString(t *testing.T) {
	feat := NewTime("blargh")
	expected := "tfeat_blargh"
	assert.Equal(t, expected, feat.String())
}

func TestTimeGet(t *testing.T) {
	feat := NewTime("blargh")

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
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			val, exists := feat.Get(td.label)
			assert.Equal(t, td.expExists, exists, "exists")
			assert.Equal(t, td.expVal, val, "value")
		})
	}
}

func TestTimeDecode(t *testing.T) {
	feat := NewTime("blargh")
	exp := map[string]string{
		"name": "blargh",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestTimeUnmarshalJSON(t *testing.T) {
	feat := NewTime("blargh")
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Time
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}

func TestTimeGenerate(t *testing.T) {
	testCases := []struct {
		name        string
		timeFeature *Time
		times       []time.Time
		expected    []float64
		tolerance   float64
		description string
	}{
		{
			name:        "unix epoch conversion",
			timeFeature: NewTime("epoch"),
			times: []time.Time{
				time.Unix(0, 0),          // 1970-01-01 00:00:00 UTC
				time.Unix(86400, 0),      // 1970-01-02 00:00:00 UTC
				time.Unix(1672531200, 0), // 2023-01-01 00:00:00 UTC
			},
			expected:  []float64{0.0, 86400.0, 1672531200.0},
			tolerance: 1e-6,
		},
		{
			name:        "nanosecond precision",
			timeFeature: NewTime("nanos"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 1, 0, 0, 0, 500000000, time.UTC), // 0.5 seconds
				time.Date(2023, 1, 1, 0, 0, 1, 0, time.UTC),         // 1 second
			},
			expected:  []float64{1672531200.0, 1672531200.5, 1672531201.0},
			tolerance: 1e-9,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.timeFeature.Generate(tc.times)

			assert.Len(t, got, len(tc.expected))

			for i, expected := range tc.expected {
				assert.InDelta(t, expected, got[i], tc.tolerance,
					"value at index %d", i)
			}
		})
	}
}

// Helper function to check for invalid float values
func isValueInvalid(val float64) bool {
	return math.IsNaN(val) || math.IsInf(val, 0)
}
