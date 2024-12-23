package forecast

import (
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/changepoint"
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

func compareFeatureSet(t *testing.T, expected, res *feature.Set, tol float64) {
	assert.Equal(t, expected.Len(), res.Len())
	require.Equal(t, expected.Labels(), res.Labels())

	for _, f := range res.Labels() {
		expVals, exists := expected.Get(f)
		require.True(t, exists)
		gotVals, exists := res.Get(f)
		require.True(t, exists)
		assert.InDeltaSlice(t, expVals, gotVals, tol, gotVals)
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("weekend"),
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
				feature.NewEvent("myevent"),
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
				feature.NewEvent("my_other_event"),
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
				feature.NewEvent("overlaps_start"),
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
				feature.NewEvent("overlaps_end"),
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
				feature.NewEvent("myevent"),
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
				feature.NewEvent("myevent"),
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
			compareFeatureSet(t, td.expected, features, 1e-4)
		})
	}
}

func TestGenerateFourierFeatures(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}

	tSeries := timedataset.GenerateT(4*7, 6*time.Hour, nowFunc)
	testData := map[string]struct {
		opt      *Options
		expected *feature.Set
		err      error
	}{
		"no seasonality": {
			opt:      &Options{},
			expected: feature.NewSet(),
			err:      nil,
		},
		"daily seasonality": {
			opt: &Options{
				DailyOrders: 2,
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("hod", feature.FourierCompSin, 1),
				[]float64{
					0, 1, 0, -1, // Thursday
					0, 1, 0, -1, // Friday
					0, 1, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, -1, // Monday
					0, 1, 0, -1, // Tuesday
					0, 1, 0, -1, // Wednesday
				},
			).Set(
				feature.NewSeasonality("hod", feature.FourierCompCos, 1),
				[]float64{
					1, 0, -1, 0, // Thursday
					1, 0, -1, 0, // Friday
					1, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, -1, 0, // Monday
					1, 0, -1, 0, // Tuesday
					1, 0, -1, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("hod", feature.FourierCompSin, 2),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("hod", feature.FourierCompCos, 2),
				[]float64{
					1, -1, 1, -1, // Thursday
					1, -1, 1, -1, // Friday
					1, -1, 1, -1, // Saturday
					1, -1, 1, -1, // Sunday
					1, -1, 1, -1, // Monday
					1, -1, 1, -1, // Tuesday
					1, -1, 1, -1, // Wednesday
				},
			),
			err: nil,
		},
		"weekly seasonality": {
			opt: &Options{
				WeeklyOrders: 2,
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("dow", feature.FourierCompSin, 1),
				[]float64{
					+0.0000, +0.2225, +0.4338, +0.6234, // Thursday
					+0.7818, +0.9009, +0.9749, +1.0000, // Friday
					+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
					+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
					-0.4338, -0.6234, -0.7818, -0.9009, // Monday
					-0.9749, -1.0000, -0.9749, -0.9009, // Tuesday
					-0.7818, -0.6234, -0.4338, -0.2225, // Wednesday
				},
			).Set(
				feature.NewSeasonality("dow", feature.FourierCompCos, 1),
				[]float64{
					+1.0000, +0.9749, +0.9009, +0.7818, // Thursday
					+0.6234, +0.4338, +0.2225, +0.0000, // Friday
					-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
					-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
					-0.9009, -0.7818, -0.6234, -0.4338, // Monday
					-0.2225, +0.0000, +0.2225, +0.4338, // Tuesday
					+0.6234, +0.7818, +0.9009, +0.9749, // Wednesday
				},
			).Set(
				feature.NewSeasonality("dow", feature.FourierCompSin, 2),
				[]float64{
					+0.0000, +0.4338, +0.7818, +0.9749, // Thursday
					+0.9749, +0.7818, +0.4338, +0.0000, // Friday
					-0.4338, -0.7818, -0.9749, -0.9749, // Saturday
					-0.7818, -0.4338, +0.0000, +0.4338, // Sunday
					+0.7818, +0.9749, +0.9749, +0.7818, // Monday
					+0.4338, +0.0000, -0.4338, -0.7818, // Tuesday
					-0.9749, -0.9749, -0.7818, -0.4338, // Wednesday
				},
			).Set(
				feature.NewSeasonality("dow", feature.FourierCompCos, 2),
				[]float64{
					+1.0000, +0.9009, +0.6234, +0.2225, // Thursday
					-0.2225, -0.6234, -0.9009, -1.0000, // Friday
					-0.9009, -0.6234, -0.2225, +0.2225, // Saturday
					+0.6234, +0.9009, +1.000, +0.9009, // Sunday
					+0.6234, +0.2225, -0.2225, -0.6234, // Monday
					-0.9009, -1.0000, -0.9009, -0.6234, // Tuesday
					-0.2225, +0.2225, +0.6234, +0.9009, // Wednesday
				},
			),
			err: nil,
		},
		"daily seasonality with weekend enabled": {
			opt: &Options{
				DailyOrders: 1,
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("hod", feature.FourierCompSin, 1),
				[]float64{
					0, 1, 0, -1, // Thursday
					0, 1, 0, -1, // Friday
					0, 1, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, -1, // Monday
					0, 1, 0, -1, // Tuesday
					0, 1, 0, -1, // Wednesday
				},
			).Set(
				feature.NewSeasonality("hod", feature.FourierCompCos, 1),
				[]float64{
					1, 0, -1, 0, // Thursday
					1, 0, -1, 0, // Friday
					1, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, -1, 0, // Monday
					1, 0, -1, 0, // Tuesday
					1, 0, -1, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_hod", feature.FourierCompSin, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_hod", feature.FourierCompCos, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					1, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
			err: nil,
		},
		"daily seasonality with event": {
			opt: &Options{
				DailyOrders: 1,
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "weekend",
							Start: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("hod", feature.FourierCompSin, 1),
				[]float64{
					0, 1, 0, -1, // Thursday
					0, 1, 0, -1, // Friday
					0, 1, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, -1, // Monday
					0, 1, 0, -1, // Tuesday
					0, 1, 0, -1, // Wednesday
				},
			).Set(
				feature.NewSeasonality("hod", feature.FourierCompCos, 1),
				[]float64{
					1, 0, -1, 0, // Thursday
					1, 0, -1, 0, // Friday
					1, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, -1, 0, // Monday
					1, 0, -1, 0, // Tuesday
					1, 0, -1, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_hod", feature.FourierCompSin, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 0, -1, // Saturday
					0, 1, 0, -1, // Sunday
					0, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_hod", feature.FourierCompCos, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					1, 0, -1, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, 0, 0, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
			err: nil,
		},
		"weekly seasonality with event": {
			opt: &Options{
				WeeklyOrders: 1,
				EventOptions: EventOptions{
					Events: []event.Event{
						{
							Name:  "weekend",
							Start: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("dow", feature.FourierCompSin, 1),
				[]float64{
					+0.0000, +0.2225, +0.4338, +0.6234, // Thursday
					+0.7818, +0.9009, +0.9749, +1.0000, // Friday
					+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
					+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
					-0.4338, -0.6234, -0.7818, -0.9009, // Monday
					-0.9749, -1.0000, -0.9749, -0.9009, // Tuesday
					-0.7818, -0.6234, -0.4338, -0.2225, // Wednesday
				},
			).Set(
				feature.NewSeasonality("dow", feature.FourierCompCos, 1),
				[]float64{
					+1.0000, +0.9749, +0.9009, +0.7818, // Thursday
					+0.6234, +0.4338, +0.2225, +0.0000, // Friday
					-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
					-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
					-0.9009, -0.7818, -0.6234, -0.4338, // Monday
					-0.2225, +0.0000, +0.2225, +0.4338, // Tuesday
					+0.6234, +0.7818, +0.9009, +0.9749, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_dow", feature.FourierCompSin, 1),
				[]float64{
					+0.0000, +0.0000, +0.0000, +0.0000, // Thursday
					+0.0000, +0.0000, +0.0000, +0.0000, // Friday
					+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
					+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
					-0.4338, +0.0000, +0.0000, +0.0000, // Monday
					+0.0000, +0.0000, +0.0000, +0.0000, // Tuesday
					+0.0000, +0.0000, +0.0000, +0.0000, // Wednesday
				},
			).Set(
				feature.NewSeasonality("weekend_dow", feature.FourierCompCos, 1),
				[]float64{
					+0.0000, +0.0000, +0.0000, +0.0000, // Thursday
					+0.0000, +0.0000, +0.0000, +0.0000, // Friday
					-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
					-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
					-0.9009, +0.0000, +0.0000, +0.0000, // Monday
					+0.0000, +0.0000, +0.0000, +0.0000, // Tuesday
					+0.0000, +0.0000, +0.0000, +0.0000, // Wednesday
				},
			),
			err: nil,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			tFeat := generateTimeFeatures(tSeries, td.opt)
			res, err := generateFourierFeatures(tFeat, td.opt)
			if td.err != nil {
				assert.EqualError(t, err, td.err.Error())
				return
			}
			compareFeatureSet(t, td.expected, res, 1e-4)
		})
	}
}

func TestGenerateChangepointFeatures(t *testing.T) {
	endTime := time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	nowFunc := func() time.Time {
		return endTime
	}

	tSeries := timedataset.GenerateT(4*7, 6*time.Hour, nowFunc)
	testData := map[string]struct {
		chpts           []changepoint.Changepoint
		trainingEndTime time.Time
		enableGrowth    bool
		expected        *feature.Set
	}{
		"no changepoints": {
			chpts:           []changepoint.Changepoint{},
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"changepoint after training end": {
			chpts: []changepoint.Changepoint{
				{Name: "chpt1", T: endTime.Add(1 * time.Minute)},
			},
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"valid single changepoint": {
			chpts: []changepoint.Changepoint{
				{Name: "chpt1", T: endTime.Add(-8 * 6 * time.Hour)},
			},
			trainingEndTime: endTime,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompBias),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					1, 1, 1, 1, // Tuesday
					1, 1, 1, 1, // Wednesday
				},
			),
		},
		"valid single changepoint with growth": {
			chpts: []changepoint.Changepoint{
				{Name: "chpt1", T: endTime.Add(-8 * 6 * time.Hour)},
			},
			trainingEndTime: endTime,
			enableGrowth:    true,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompBias),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					1, 1, 1, 1, // Tuesday
					1, 1, 1, 1, // Wednesday
				},
			).Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompSlope),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0.000, 0.125, 0.250, 0.375, // Tuesday
					0.500, 0.625, 0.750, 0.875, // Wednesday
				},
			),
		},
		"valid single changepoint with growth and future training date": {
			chpts: []changepoint.Changepoint{
				{Name: "chpt1", T: endTime.Add(-8 * 6 * time.Hour)},
			},
			trainingEndTime: endTime.Add(8 * 6 * time.Hour),
			enableGrowth:    true,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompBias),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					1, 1, 1, 1, // Tuesday
					1, 1, 1, 1, // Wednesday
				},
			).Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompSlope),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0, 0, 0, 0, // Monday
					0.0000, 0.0625, 0.1250, 0.1875, // Tuesday
					0.2500, 0.3125, 0.3750, 0.4375, // Wednesday
				},
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := generateChangepointFeatures(tSeries, td.chpts, td.trainingEndTime, td.enableGrowth)
			compareFeatureSet(t, td.expected, res, 1e-4)
		})
	}
}
