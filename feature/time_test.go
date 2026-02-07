package feature

import (
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
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)

	testCases := []struct {
		name        string
		timeFeature *Time
		times       []time.Time
		wantLen     int
		description string
	}{
		{
			name:        "basic time series",
			timeFeature: NewTime("epoch"),
			times: []time.Time{
				baseTime,
				baseTime.Add(1 * time.Hour),
				baseTime.Add(2 * time.Hour),
				baseTime.Add(3 * time.Hour),
			},
			wantLen:     4,
			description: "Basic time series with hourly increments",
		},
		{
			name:        "single time point",
			timeFeature: NewTime("single"),
			times: []time.Time{
				baseTime,
			},
			wantLen:     1,
			description: "Single time point",
		},
		{
			name:        "empty time series",
			timeFeature: NewTime("empty"),
			times:       []time.Time{},
			wantLen:     0,
			description: "Empty input time series",
		},
		{
			name:        "daily time series",
			timeFeature: NewTime("daily"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC),
			},
			wantLen:     5,
			description: "Daily time series",
		},
		{
			name:        "time series with milliseconds",
			timeFeature: NewTime("millis"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 100000000, time.UTC), // 0.1 seconds
				time.Date(2023, 1, 1, 0, 0, 0, 200000000, time.UTC), // 0.2 seconds
				time.Date(2023, 1, 1, 0, 0, 0, 300000000, time.UTC), // 0.3 seconds
			},
			wantLen:     3,
			description: "Time series with millisecond precision",
		},
		{
			name:        "time series with nanoseconds",
			timeFeature: NewTime("nanos"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 1, time.UTC),   // 1 nanosecond
				time.Date(2023, 1, 1, 0, 0, 0, 50, time.UTC),  // 50 nanoseconds
				time.Date(2023, 1, 1, 0, 0, 0, 100, time.UTC), // 100 nanoseconds
			},
			wantLen:     3,
			description: "Time series with nanosecond precision",
		},
		{
			name:        "time series across years",
			timeFeature: NewTime("years"),
			times: []time.Time{
				time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2021, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2022, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
			},
			wantLen:     4,
			description: "Time series spanning multiple years",
		},
		{
			name:        "time series with timezones",
			timeFeature: NewTime("timezone"),
			times: []time.Time{
				time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 1, 12, 0, 0, 0, time.FixedZone("EST", -5*3600)),
				time.Date(2023, 1, 1, 12, 0, 0, 0, time.FixedZone("JST", 9*3600)),
			},
			wantLen:     3,
			description: "Time series with different timezones",
		},
		{
			name:        "monotonically increasing times",
			timeFeature: NewTime("monotonic"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 3, 6, 0, 0, 0, time.UTC),
			},
			wantLen:     4,
			description: "Monotonically increasing time series",
		},
		{
			name:        "unix epoch time",
			timeFeature: NewTime("unix_epoch"),
			times: []time.Time{
				time.Unix(0, 0),          // Unix epoch
				time.Unix(86400, 0),      // 1 day later
				time.Unix(86400*365, 0),  // 1 year later
				time.Unix(1672531200, 0), // 2023-01-01
			},
			wantLen:     4,
			description: "Times starting from Unix epoch",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.timeFeature.Generate(tc.times)

			assert.Len(t, got, tc.wantLen)

			// Validate output values
			for i, val := range got {
				assert.False(t, isValueInvalid(val), "value at index %d should not be NaN or infinite", i)
				// Unix epoch can be 0 for the very first moment
				assert.True(t, val >= 0, "epoch value at index %d should be non-negative", i)
			}

			// Validate monotonicity for sorted input times
			if len(got) > 1 {
				for i := 1; i < len(got); i++ {
					prevTime := tc.times[i-1]
					currTime := tc.times[i]

					if currTime.After(prevTime) || currTime.Equal(prevTime) {
						assert.True(t, got[i] >= got[i-1],
							"epoch values should be non-decreasing when time is non-decreasing (index %d: %f >= %f)",
							i, got[i], got[i-1])
					}
				}
			}
		})
	}
}

func TestTimeGenerateKnownValues(t *testing.T) {
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
			expected:    []float64{0.0, 86400.0, 1672531200.0},
			tolerance:   1e-6,
			description: "Direct Unix timestamp conversion",
		},
		{
			name:        "nanosecond precision",
			timeFeature: NewTime("nanos"),
			times: []time.Time{
				time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC),
				time.Date(2023, 1, 1, 0, 0, 0, 500000000, time.UTC), // 0.5 seconds
				time.Date(2023, 1, 1, 0, 0, 1, 0, time.UTC),         // 1 second
			},
			expected:    []float64{1672531200.0, 1672531200.5, 1672531201.0},
			tolerance:   1e-9,
			description: "Nanosecond precision timestamps",
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
	return isValueNaN(val) || isValueInf(val)
}

func isValueNaN(val float64) bool {
	return val != val // NaN is the only value that is not equal to itself
}

func isValueInf(val float64) bool {
	return val > 1.7976931348623157e+308 || val < -1.7976931348623157e+308
}
