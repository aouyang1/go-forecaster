package forecast

import (
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/event"
	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGetLocationDSTOffset(t *testing.T) {
	testData := map[string]struct {
		name     string
		err      error
		expected int
	}{
		"northern hemisphere": {"America/Los_Angeles", nil, 3600},
		"southern hemisphere": {"Australia/South", nil, 3600},
		"30min offset":        {"Australia/LHI", nil, 1800},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			loc, err := time.LoadLocation(td.name)
			require.Nil(t, err)
			offset := getLocationDSTOffset(loc)
			if td.err != nil {
				assert.ErrorContains(t, err, td.err.Error())
				return
			}

			require.Nil(t, err)
			assert.Equal(t, td.expected, offset)
		})
	}
}

func TestAdjustTime(t *testing.T) {
	testData := map[string]struct {
		input    time.Time
		zoneLoc  []string
		expected time.Time
	}{
		"america dst pre-std fall": {
			time.Date(2024, time.November, 3, 8, 59, 59, 0, time.UTC), // 2024-11-03 01:59:59 PST
			[]string{"America/Los_Angeles"},
			time.Date(2024, time.November, 3, 9, 59, 59, 0, time.UTC),
		},
		"america to std fall": {
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC), // 2024-11-03 02:00:00 PST
			[]string{"America/Los_Angeles"},
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC),
		},
		"america std pre-dst spring": {
			time.Date(2025, time.March, 9, 9, 59, 59, 0, time.UTC), // 2024-03-09 01:59:59 PST
			[]string{"America/Los_Angeles"},
			time.Date(2025, time.March, 9, 9, 59, 59, 0, time.UTC),
		},
		"america to dst spring": {
			time.Date(2025, time.March, 9, 10, 0, 0, 0, time.UTC), // 2024-03-09 02:00:00 PST
			[]string{"America/Los_Angeles"},
			time.Date(2025, time.March, 9, 11, 0, 0, 0, time.UTC),
		},
		"europe dst pre-std fall": {
			time.Date(2024, time.October, 27, 0, 59, 59, 0, time.UTC), // 2024-10-27 00:59:59
			[]string{"Europe/London"},
			time.Date(2024, time.October, 27, 1, 59, 59, 0, time.UTC),
		},
		"europe to std fall": {
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC), // 2024-11-03 01:00:00
			[]string{"Europe/London"},
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC),
		},
		"europe dst pre-std america dst fall": {
			time.Date(2024, time.October, 27, 0, 59, 59, 0, time.UTC), // 2024-10-27 00:59:59
			[]string{"Europe/London", "America/Los_Angeles"},
			time.Date(2024, time.October, 27, 1, 59, 59, 0, time.UTC),
		},
		"europe to std america dst fall": {
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC),
			[]string{"Europe/London", "America/Los_Angeles"},
			time.Date(2024, time.October, 27, 1, 30, 0, 0, time.UTC),
		},
		"europe std america dst pre-std fall": {
			time.Date(2024, time.November, 3, 8, 59, 59, 0, time.UTC),
			[]string{"Europe/London", "America/Los_Angeles"},
			time.Date(2024, time.November, 3, 9, 29, 59, 0, time.UTC),
		},
		"europe std america std fall": {
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC),
			[]string{"Europe/London", "America/Los_Angeles"},
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			offsets := loadLocationOffsets(td.zoneLoc)
			res := adjustTime(td.input, offsets)
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestGenerateTimeFeatures(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}
	testData := map[string]struct {
		t        []time.Time
		opt      *Options
		expected *feature.Set
	}{
		"empty options": {
			t:        timedataset.GenerateT(24*7, time.Hour, nowFunc),
			opt:      &Options{},
			expected: feature.NewSet(),
		},
		"basic weekend": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: 6 * time.Hour,
					DurAfter:  12 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 1, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					1, 1, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with max extension buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: 30 * time.Hour,
					DurAfter:  30 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					1, 1, 1, 1, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 1, // Sunday
					1, 1, 1, 1, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with max shrink buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: -30 * time.Hour,
					DurAfter:  -30 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with shift buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: 6 * time.Hour,
					DurAfter:  -6 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 1, // Friday
					1, 1, 1, 1, // Saturday
					1, 1, 1, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},

		"weekend with tz override": {
			t: timedataset.GenerateT(3*7, 8*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:          true,
					TimezoneOverride: "America/Los_Angeles",
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, // Thursday
					0, 0, 0, // Friday
					0, 1, 1, // Saturday
					1, 1, 1, // Sunday
					1, 0, 0, // Monday
					0, 0, 0, // Tuesday
					0, 0, 0, // Wednesday
				},
			),
		},
		"weekend with invalid tz override": {
			t: timedataset.GenerateT(3*7, 8*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:          true,
					TimezoneOverride: "bogus",
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, // Thursday
					0, 0, 0, // Friday
					1, 1, 1, // Saturday
					1, 1, 1, // Sunday
					0, 0, 0, // Monday
					0, 0, 0, // Tuesday
					0, 0, 0, // Wednesday
				},
			),
		},
		"basic weekend with hamm window": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("is_weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0.1882, 0.6112, 0.9504, // Saturday
					0.9504, 0.6112, 0.1882, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"basic event": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 1, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "my other event",
							Start: time.Date(1970, 1, 7, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("event_myevent"),
				[]float64{
					0, 1, 1, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewTime("event_my_other_event"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 1, 1, 0, // Wednesday
				},
			),
		},
		"event extending out of bounds": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "overlaps_start",
							Start: time.Date(1969, 12, 30, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "overlaps_end",
							Start: time.Date(1970, 1, 7, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 9, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("event_overlaps_start"),
				[]float64{
					1, 1, 1, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewTime("event_overlaps_end"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 1, 1, 1, // Wednesday
				},
			),
		},
		"duplicate event": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 1, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 7, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("event_myevent"),
				[]float64{
					0, 1, 1, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"invalid events": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name: "start is zero",
							End:  time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "end is zero",
							Start: time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
						{
							Start: time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "start after end",
							Start: time.Date(1970, 1, 7, 12, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet(),
		},
		"basic event with hamm window": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 2, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("event_myevent"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0.0000, 0.0000, 0.1882, 0.6112, // Friday
					0.9504, 0.9504, 0.6112, 0.1882, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
		},
		"daily orders": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				DailyOrders: 2,
			},
			expected: feature.NewSet().Set(
				feature.NewTime("hod"),
				[]float64{
					0, 6, 12, 18, // Thursday
					0, 6, 12, 18, // Friday
					0, 6, 12, 18, // Saturday
					0, 6, 12, 18, // Sunday
					0, 6, 12, 18, // Monday
					0, 6, 12, 18, // Tuesday
					0, 6, 12, 18, // Wednesday
				},
			),
		},
		"weekly orders": {
			t: timedataset.GenerateT(2*14, 12*time.Hour, nowFunc),
			opt: &Options{
				WeeklyOrders: 2,
			},
			expected: feature.NewSet().Set(
				feature.NewTime("dow"),
				[]float64{
					0, -6.50, // Thursday
					-6, -5.50, // Friday
					-5, -4.50, // Saturday
					-4, -3.50, // Sunday
					-3, -2.50, // Monday
					-2, -1.50, // Tuesday
					-1, -0.50, // Wednesday
					0, 0.50, // Thursday
					1, 1.50, // Friday
					2, 2.50, // Saturday
					3, 3.50, // Sunday
					4, 4.50, // Monday
					5, 5.50, // Tuesday
					6, 6.50, // Wednesday
				},
			),
		},
		"default": {
			t:   timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: nil,
			expected: feature.NewSet().Set(
				feature.NewTime("hod"),
				[]float64{
					0, 6, 12, 18, // Thursday
					0, 6, 12, 18, // Friday
					0, 6, 12, 18, // Saturday
					0, 6, 12, 18, // Sunday
					0, 6, 12, 18, // Monday
					0, 6, 12, 18, // Tuesday
					0, 6, 12, 18, // Wednesday
				},
			).Set(
				feature.NewTime("dow"),
				[]float64{
					0, 0.25, 0.50, 0.75, // Thursday
					1, 1.25, 1.50, 1.75, // Friday
					2, 2.25, 2.50, 2.75, // Saturday
					3, 3.25, 3.50, 3.75, // Sunday
					4, 4.25, 4.50, 4.75, // Monday
					5, 5.25, 5.50, 5.75, // Tuesday
					6, 6.25, 6.50, 6.75, // Wednesday
				},
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			features := generateTimeFeatures(td.t, td.opt)
			assert.Equal(t, td.expected.Len(), features.Len())
			require.Equal(t, td.expected.Labels(), features.Labels())

			for _, f := range features.Labels() {
				expVals, exists := td.expected.Get(f)
				require.True(t, exists)
				gotVals, exists := features.Get(f)
				require.True(t, exists)
				assert.InDeltaSlice(t, expVals, gotVals, 1e-4, gotVals)
			}
		})
	}
}
