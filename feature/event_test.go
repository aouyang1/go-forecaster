package feature

import (
	"testing"
	"time"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/dsp/window"
	"gonum.org/v1/gonum/floats"
)

func TestEventString(t *testing.T) {
	feat := NewEvent("blargh")
	expected := "event_blargh"
	assert.Equal(t, expected, feat.String())
}

func TestEventGet(t *testing.T) {
	feat := NewEvent("blargh")

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

func TestEventDecode(t *testing.T) {
	feat := NewEvent("blargh")
	exp := map[string]string{
		"name": "blargh",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestEventUnmarshalJSON(t *testing.T) {
	feat := NewEvent("blargh")
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Event
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}

func TestEventStandardGenerate(t *testing.T) {
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)

	testCases := []struct {
		name        string
		event       EventStandard
		times       []time.Time
		window      string
		wantLen     int
		wantErr     bool
		errContains string
		description string
	}{
		{
			name: "basic event within time range",
			event: EventStandard{
				Start: time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour), baseTime.Add(72 * time.Hour), baseTime.Add(96 * time.Hour)},
			window:      WindowRectangular,
			wantLen:     5,
			wantErr:     false,
			description: "Event spans middle 3 days of 5-day series",
		},
		{
			name: "event outside time range",
			event: EventStandard{
				Start: time.Date(2023, 1, 10, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 12, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour)},
			window:      WindowRectangular,
			wantLen:     3,
			wantErr:     false,
			description: "Event completely outside time range",
		},
		{
			name: "event partially overlaps start",
			event: EventStandard{
				Start: time.Date(2022, 12, 31, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour)},
			window:      WindowRectangular,
			wantLen:     3,
			wantErr:     false,
			description: "Event starts before and overlaps first point",
		},
		{
			name: "event partially overlaps end",
			event: EventStandard{
				Start: time.Date(2023, 1, 3, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 5, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour)},
			window:      WindowRectangular,
			wantLen:     3,
			wantErr:     false,
			description: "Event ends after last time point",
		},
		{
			name: "empty time series",
			event: EventStandard{
				Start: time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{},
			window:      WindowRectangular,
			wantLen:     0,
			wantErr:     true,
			errContains: "cannot infer frequency",
			description: "Empty input time series should error",
		},
		{
			name: "single point time series",
			event: EventStandard{
				Start: time.Date(2023, 1, 1, 12, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 1, 14, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime},
			window:      WindowRectangular,
			wantLen:     0,
			wantErr:     true,
			errContains: "cannot infer frequency",
			description: "Single time point should error",
		},
		{
			name: "regular time series with event",
			event: EventStandard{
				Start: time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
			},
			times: []time.Time{
				baseTime,
				baseTime.Add(24 * time.Hour),
				baseTime.Add(48 * time.Hour),
				baseTime.Add(72 * time.Hour),
				baseTime.Add(96 * time.Hour),
			},
			window:      WindowRectangular,
			wantLen:     5,
			wantErr:     false,
			description: "Regular time series with overlapping event",
		},
		{
			name: "event with hann window",
			event: EventStandard{
				Start: time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour), baseTime.Add(72 * time.Hour), baseTime.Add(96 * time.Hour)},
			window:      WindowHann,
			wantLen:     5,
			wantErr:     false,
			description: "Event with Hann windowing function",
		},
		{
			name: "event with hamming window",
			event: EventStandard{
				Start: time.Date(2023, 1, 2, 0, 0, 0, 0, time.UTC),
				End:   time.Date(2023, 1, 4, 0, 0, 0, 0, time.UTC),
			},
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour), baseTime.Add(72 * time.Hour), baseTime.Add(96 * time.Hour)},
			window:      WindowHamming,
			wantLen:     5,
			wantErr:     false,
			description: "Event with Hamming windowing function",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.event.Generate(tc.times, tc.window)

			if tc.wantErr {
				assert.Error(t, err)
				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains)
				}
				return
			}

			assert.NoError(t, err)
			assert.NotNil(t, got)
			assert.Len(t, got, tc.wantLen)

			// Additional validation for non-empty results
			if len(got) > 0 && tc.window == WindowRectangular {
				// For rectangular window, values should be exactly 0 or 1
				for i, val := range got {
					assert.True(t, val == 0.0 || val == 1.0,
						"value at index %d should be 0 or 1, got %f", i, val)
				}
			}
		})
	}
}

func TestEventWeekendGenerate(t *testing.T) {
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC) // Sunday

	testCases := []struct {
		name        string
		event       EventWeekend
		times       []time.Time
		window      string
		wantLen     int
		wantErr     bool
		description string
	}{
		{
			name: "basic weekend detection",
			event: EventWeekend{
				DurBefore: 0,
				DurAfter:  0,
			},
			times: []time.Time{
				baseTime,                      // Sunday (weekend)
				baseTime.Add(24 * time.Hour),  // Monday
				baseTime.Add(48 * time.Hour),  // Tuesday
				baseTime.Add(72 * time.Hour),  // Wednesday
				baseTime.Add(96 * time.Hour),  // Thursday
				baseTime.Add(120 * time.Hour), // Friday
				baseTime.Add(144 * time.Hour), // Saturday (weekend)
				baseTime.Add(168 * time.Hour), // Sunday (weekend)
				baseTime.Add(192 * time.Hour), // Monday
			},
			window:      WindowRectangular,
			wantLen:     9,
			wantErr:     false,
			description: "Detect weekends correctly without buffer",
		},
		{
			name: "weekend with positive buffer",
			event: EventWeekend{
				DurBefore: 2 * time.Hour,
				DurAfter:  2 * time.Hour,
			},
			times: []time.Time{
				baseTime.Add(22 * time.Hour), // Saturday 22:00 (within buffer)
				baseTime.Add(24 * time.Hour), // Sunday 00:00 (weekend)
				baseTime.Add(48 * time.Hour), // Monday
			},
			window:      WindowRectangular,
			wantLen:     3,
			wantErr:     false,
			description: "Weekend with 2-hour buffer before and after",
		},
		{
			name: "weekend with negative buffer",
			event: EventWeekend{
				DurBefore: -2 * time.Hour,
				DurAfter:  -2 * time.Hour,
			},
			times: []time.Time{
				baseTime.Add(2 * time.Hour),  // Sunday 02:00 (weekend)
				baseTime.Add(24 * time.Hour), // Monday
			},
			window:      WindowRectangular,
			wantLen:     2,
			wantErr:     false,
			description: "Weekend with negative buffer (strict hours)",
		},
		{
			name: "weekend with timezone override",
			event: EventWeekend{
				TimezoneOverride: "America/New_York",
				DurBefore:        0,
				DurAfter:         0,
			},
			times: []time.Time{
				baseTime,                     // Sunday UTC
				baseTime.Add(24 * time.Hour), // Monday UTC
			},
			window:      WindowRectangular,
			wantLen:     2,
			wantErr:     false,
			description: "Weekend with timezone override",
		},
		{
			name: "weekend with invalid timezone",
			event: EventWeekend{
				TimezoneOverride: "Invalid/Timezone",
				DurBefore:        0,
				DurAfter:         0,
			},
			times: []time.Time{
				baseTime,
				baseTime.Add(24 * time.Hour),
			},
			window:      WindowRectangular,
			wantLen:     2,
			wantErr:     false,
			description: "Weekend with invalid timezone should fallback gracefully",
		},
		{
			name: "weekend with extreme buffer values",
			event: EventWeekend{
				DurBefore: 48 * time.Hour, // Should be capped to MaxWeekendDurBuffer
				DurAfter:  48 * time.Hour,
			},
			times: []time.Time{
				baseTime.Add(-48 * time.Hour), // Previous Thursday
				baseTime.Add(24 * time.Hour),  // Monday
			},
			window:      WindowRectangular,
			wantLen:     2,
			wantErr:     false,
			description: "Weekend with extreme buffer values should be capped",
		},
		{
			name: "weekend with hann window",
			event: EventWeekend{
				DurBefore: 0,
				DurAfter:  0,
			},
			times: []time.Time{
				baseTime,                      // Sunday
				baseTime.Add(24 * time.Hour),  // Monday
				baseTime.Add(48 * time.Hour),  // Tuesday
				baseTime.Add(120 * time.Hour), // Friday
				baseTime.Add(144 * time.Hour), // Saturday
			},
			window:      WindowHann,
			wantLen:     5,
			wantErr:     false,
			description: "Weekend with Hann windowing function",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.event.Validate() // Ensure buffer limits are applied
			got, err := tc.event.Generate(tc.times, tc.window)

			if tc.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Len(t, got, tc.wantLen)

			// Validate output values are within expected range
			for i, val := range got {
				assert.True(t, val >= 0.0 && val <= 1.0,
					"value at index %d should be between 0 and 1, got %f", i, val)
			}
		})
	}
}

func TestEventGenerate(t *testing.T) {
	baseTime := time.Date(2023, 1, 1, 0, 0, 0, 0, time.UTC)

	testCases := []struct {
		name        string
		event       *Event
		times       []time.Time
		window      string
		evGenerator EventGenerator
		wantLen     int
		wantErr     bool
		description string
	}{
		{
			name:        "Event with standard setting",
			event:       NewEvent("test"),
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour), baseTime.Add(48 * time.Hour)},
			window:      WindowRectangular,
			evGenerator: &EventStandard{Start: baseTime.Add(12 * time.Hour), End: baseTime.Add(36 * time.Hour)},
			wantLen:     3,
			wantErr:     false,
			description: "Event using standard setting",
		},
		{
			name:        "Event with weekend setting",
			event:       NewEvent("weekend_test"),
			times:       []time.Time{baseTime, baseTime.Add(24 * time.Hour)},
			window:      WindowRectangular,
			evGenerator: &EventWeekend{DurBefore: 0, DurAfter: 0},
			wantLen:     2,
			wantErr:     false,
			description: "Event using weekend setting",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if weekend, ok := tc.evGenerator.(*EventWeekend); ok {
				weekend.Validate()
			}

			got, err := tc.event.Generate(tc.times, tc.window, tc.evGenerator)

			if tc.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.Len(t, got, tc.wantLen)
		})
	}
}

func TestWindowFunc(t *testing.T) {
	testData := map[string]struct {
		name     string
		expected func([]float64) []float64
	}{
		"bartlett hann":    {WindowBartlettHann, window.BartlettHann},
		"blackman":         {WindowBlackman, window.Blackman},
		"blackman harris":  {WindowBlackmanHarris, window.BlackmanHarris},
		"blackman nuttall": {WindowBlackmanNuttall, window.BlackmanNuttall},
		"flat top":         {WindowFlatTop, window.FlatTop},
		"hamming":          {WindowHamming, window.Hamming},
		"hann":             {WindowHann, window.Hann},
		"lanczos":          {WindowLanczos, window.Lanczos},
		"nuttall":          {WindowNuttall, window.Nuttall},
		"rectangular":      {WindowRectangular, window.Rectangular},
		"sine":             {WindowSine, window.Sine},
		"triangular":       {WindowTriangular, window.Triangular},
		"unknown":          {"unknown", window.Rectangular},
	}

	numPnts := 10
	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := WindowFunc(td.name)

			seqRes := make([]float64, numPnts)
			floats.AddConst(1.0, seqRes)

			seqExp := make([]float64, numPnts)
			floats.AddConst(1.0, seqExp)

			assert.Equal(t, td.expected(seqExp), res(seqRes))
		})
	}
}
