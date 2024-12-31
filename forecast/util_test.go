package forecast

import (
	"fmt"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	TZAmericaLosAngeles = "America/Los_Angeles"
	TZEuropeLondon      = "Europe/London"
)

func TestGetLocationDSTOffset(t *testing.T) {
	testData := map[string]struct {
		name     string
		err      error
		expected int
	}{
		"northern hemisphere": {TZAmericaLosAngeles, nil, 3600},
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
			[]string{TZAmericaLosAngeles},
			time.Date(2024, time.November, 3, 9, 59, 59, 0, time.UTC),
		},
		"america to std fall": {
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC), // 2024-11-03 02:00:00 PST
			[]string{TZAmericaLosAngeles},
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC),
		},
		"america std pre-dst spring": {
			time.Date(2025, time.March, 9, 9, 59, 59, 0, time.UTC), // 2024-03-09 01:59:59 PST
			[]string{TZAmericaLosAngeles},
			time.Date(2025, time.March, 9, 9, 59, 59, 0, time.UTC),
		},
		"america to dst spring": {
			time.Date(2025, time.March, 9, 10, 0, 0, 0, time.UTC), // 2024-03-09 02:00:00 PST
			[]string{TZAmericaLosAngeles},
			time.Date(2025, time.March, 9, 11, 0, 0, 0, time.UTC),
		},
		"europe dst pre-std fall": {
			time.Date(2024, time.October, 27, 0, 59, 59, 0, time.UTC), // 2024-10-27 00:59:59
			[]string{TZEuropeLondon},
			time.Date(2024, time.October, 27, 1, 59, 59, 0, time.UTC),
		},
		"europe to std fall": {
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC), // 2024-11-03 01:00:00
			[]string{TZEuropeLondon},
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC),
		},
		"europe dst pre-std america dst fall": {
			time.Date(2024, time.October, 27, 0, 59, 59, 0, time.UTC), // 2024-10-27 00:59:59
			[]string{TZEuropeLondon, TZAmericaLosAngeles},
			time.Date(2024, time.October, 27, 1, 59, 59, 0, time.UTC),
		},
		"europe to std america dst fall": {
			time.Date(2024, time.October, 27, 1, 0, 0, 0, time.UTC),
			[]string{TZEuropeLondon, TZAmericaLosAngeles},
			time.Date(2024, time.October, 27, 1, 30, 0, 0, time.UTC),
		},
		"europe std america dst pre-std fall": {
			time.Date(2024, time.November, 3, 8, 59, 59, 0, time.UTC),
			[]string{TZEuropeLondon, TZAmericaLosAngeles},
			time.Date(2024, time.November, 3, 9, 29, 59, 0, time.UTC),
		},
		"europe std america std fall": {
			time.Date(2024, time.November, 3, 9, 0, 0, 0, time.UTC),
			[]string{TZEuropeLondon, TZAmericaLosAngeles},
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
		assert.InDeltaSlice(t, expVals, gotVals, tol, fmt.Sprintf("feature: %+v, values: %+v\n", f, gotVals))
	}
}

var (
	epoch7DaysAt6Hr = []float64{
		0 * 3600.0, 6 * 3600.0, 12 * 3600.0, 18 * 3600.0, // Thursday
		24 * 3600, 30 * 3600.0, 36 * 3600.0, 42 * 3600.0, // Friday
		48 * 3600, 54 * 3600.0, 60 * 3600.0, 66 * 3600.0, // Saturday
		72 * 3600, 78 * 3600.0, 84 * 3600.0, 90 * 3600.0, // Sunday
		96 * 3600, 102 * 3600.0, 108 * 3600.0, 114 * 3600.0, // Monday
		120 * 3600, 126 * 3600.0, 132 * 3600.0, 138 * 3600.0, // Tuesday
		144 * 3600, 150 * 3600.0, 156 * 3600.0, 162 * 3600.0, // Wednesday
	}
	epoch7DaysAt8Hr = []float64{
		0 * 3600.0, 8 * 3600.0, 16 * 3600.0, // Thursday
		24 * 3600, 32 * 3600.0, 40 * 3600.0, // Friday
		48 * 3600, 56 * 3600.0, 64 * 3600.0, // Saturday
		72 * 3600, 80 * 3600.0, 88 * 3600.0, // Sunday
		96 * 3600, 104 * 3600.0, 112 * 3600.0, // Monday
		120 * 3600, 128 * 3600.0, 136 * 3600.0, // Tuesday
		144 * 3600, 152 * 3600.0, 160 * 3600.0, // Wednesday
	}
)

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
			t:   timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{},
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
		"basic weekend": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
		"weekend with shrink buffers": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				WeekendOptions: WeekendOptions{
					Enabled:   true,
					DurBefore: -6 * time.Hour,
					DurAfter:  -6 * time.Hour,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("weekend"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 1, 1, // Saturday
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
				feature.NewTime("epoch"),
				epoch7DaysAt8Hr,
			).Set(
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
				feature.NewTime("epoch"),
				epoch7DaysAt8Hr,
			).Set(
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
					Events: []Event{
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
					Events: []Event{
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
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
					Events: []Event{
						{
							Name:  "duplicate",
							Start: time.Date(1970, 1, 3, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 3, 12, 0, 0, 0, time.UTC),
						},
						{
							Name:  "duplicate",
							Start: time.Date(1970, 1, 2, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 2, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
				feature.NewEvent("duplicate"),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 1, 1, 0, // Saturday
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
					Events: []Event{
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
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
		"basic event with hamm window": {
			t: timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: &Options{
				MaskWindow: "hann",
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 2, 6, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			).Set(
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
		"default": {
			t:   timedataset.GenerateT(4*7, 6*time.Hour, nowFunc),
			opt: nil,
			expected: feature.NewSet().Set(
				feature.NewTime("epoch"),
				epoch7DaysAt6Hr,
			),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			features, _ := generateTimeFeatures(td.t, td.opt)
			compareFeatureSet(t, td.expected, features, 1e-4)
		})
	}
}

func TestGenerateFourierFeatures(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}

	dailySin1 := []float64{
		0, 1, 0, -1, // Thursday
		0, 1, 0, -1, // Friday
		0, 1, 0, -1, // Saturday
		0, 1, 0, -1, // Sunday
		0, 1, 0, -1, // Monday
		0, 1, 0, -1, // Tuesday
		0, 1, 0, -1, // Wednesday
	}
	dailyCos1 := []float64{
		1, 0, -1, 0, // Thursday
		1, 0, -1, 0, // Friday
		1, 0, -1, 0, // Saturday
		1, 0, -1, 0, // Sunday
		1, 0, -1, 0, // Monday
		1, 0, -1, 0, // Tuesday
		1, 0, -1, 0, // Wednesday
	}
	dailySin2 := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		0, 0, 0, 0, // Saturday
		0, 0, 0, 0, // Sunday
		0, 0, 0, 0, // Monday
		0, 0, 0, 0, // Tuesday
		0, 0, 0, 0, // Wednesday
	}
	dailyCos2 := []float64{
		1, -1, 1, -1, // Thursday
		1, -1, 1, -1, // Friday
		1, -1, 1, -1, // Saturday
		1, -1, 1, -1, // Sunday
		1, -1, 1, -1, // Monday
		1, -1, 1, -1, // Tuesday
		1, -1, 1, -1, // Wednesday
	}
	weeklySin1 := []float64{
		+0.0000, +0.2225, +0.4338, +0.6234, // Thursday
		+0.7818, +0.9009, +0.9749, +1.0000, // Friday
		+0.9749, +0.9009, +0.7818, +0.6234, // Saturday
		+0.4338, +0.2225, +0.0000, -0.2225, // Sunday
		-0.4338, -0.6234, -0.7818, -0.9009, // Monday
		-0.9749, -1.0000, -0.9749, -0.9009, // Tuesday
		-0.7818, -0.6234, -0.4338, -0.2225, // Wednesday
	}
	weeklyCos1 := []float64{
		+1.0000, +0.9749, +0.9009, +0.7818, // Thursday
		+0.6234, +0.4338, +0.2225, +0.0000, // Friday
		-0.2225, -0.4338, -0.6234, -0.7818, // Saturday
		-0.9009, -0.9749, -1.0000, -0.9749, // Sunday
		-0.9009, -0.7818, -0.6234, -0.4338, // Monday
		-0.2225, +0.0000, +0.2225, +0.4338, // Tuesday
		+0.6234, +0.7818, +0.9009, +0.9749, // Wednesday
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
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(2),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 2),
				dailySin2,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 2),
				dailyCos2,
			),
			err: nil,
		},
		"weekly seasonality": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewWeeklySeasonalityConfig(2),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 2),
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
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 2),
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
		"duplicate seasonality": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
						NewDailySeasonalityConfig(1),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			),
			err: nil,
		},

		"daily seasonality with weekend enabled": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
					},
				},
				WeekendOptions: WeekendOptions{
					Enabled: true,
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("weekend_daily", feature.FourierCompSin, 1),
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
				feature.NewSeasonality("weekend_daily", feature.FourierCompCos, 1),
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
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewDailySeasonalityConfig(1),
					},
				},
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "myevent",
							Start: time.Date(1970, 1, 4, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 6, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompSin, 1),
				dailySin1,
			).Set(
				feature.NewSeasonality("epoch_daily", feature.FourierCompCos, 1),
				dailyCos1,
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompSin, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 1, 0, -1, // Sunday
					0, 1, 0, -1, // Monday
					0, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			).Set(
				feature.NewSeasonality("myevent_daily", feature.FourierCompCos, 1),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					1, 0, -1, 0, // Sunday
					1, 0, -1, 0, // Monday
					1, 0, 0, 0, // Tuesday
					0, 0, 0, 0, // Wednesday
				},
			),
			err: nil,
		},
		"weekly seasonality with event": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						NewWeeklySeasonalityConfig(1),
					},
				},
				EventOptions: EventOptions{
					Events: []Event{
						{
							Name:  "weekend",
							Start: time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
							End:   time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("weekend_weekly", feature.FourierCompSin, 1),
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
				feature.NewSeasonality("weekend_weekly", feature.FourierCompCos, 1),
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
		"weekly seasonality with colinear": {
			opt: &Options{
				SeasonalityOptions: SeasonalityOptions{
					SeasonalityConfigs: []SeasonalityConfig{
						{
							Name:   "colinear",
							Period: 3.5 * 24 * time.Hour,
							Orders: 2,
						},
						NewWeeklySeasonalityConfig(4),
					},
				},
			},
			expected: feature.NewSet().Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompSin, 1),
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
				feature.NewSeasonality("epoch_colinear", feature.FourierCompCos, 1),
				[]float64{
					+1.0000, +0.9009, +0.6234, +0.2225, // Thursday
					-0.2225, -0.6234, -0.9009, -1.0000, // Friday
					-0.9009, -0.6234, -0.2225, +0.2225, // Saturday
					+0.6234, +0.9009, +1.0000, +0.9009, // Sunday
					+0.6234, +0.2225, -0.2225, -0.6234, // Monday
					-0.9009, -1.0000, -0.9009, -0.6234, // Tuesday
					-0.2225, +0.2225, +0.6234, +0.9009, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompSin, 2),
				[]float64{
					+0.0000, +0.7818, +0.9749, +0.4338, // Thursday
					-0.4338, -0.9749, -0.7818, +0.0000, // Friday
					+0.7818, +0.9749, +0.4338, -0.4338, // Saturday
					-0.9749, -0.7818, +0.0000, +0.7818, // Sunday
					+0.9749, +0.4338, -0.4338, -0.9749, // Monday
					-0.7818, +0.0000, +0.7818, +0.9749, // Tuesday
					+0.4338, -0.4338, -0.9749, -0.7818, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_colinear", feature.FourierCompCos, 2),
				[]float64{
					+1.0000, +0.6234, -0.2225, -0.9009, // Thursday
					-0.9009, -0.2225, +0.6234, +1.0000, // Friday
					+0.6234, -0.2225, -0.9009, -0.9009, // Saturday
					-0.2225, +0.6234, +1.0000, +0.6234, // Sunday
					-0.2225, -0.9009, -0.9009, -0.2225, // Monday
					+0.6234, +1.0000, +0.6234, -0.2225, // Tuesday
					-0.9009, -0.9009, -0.2225, +0.6234, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 1),
				weeklySin1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 1),
				weeklyCos1,
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompSin, 3),
				[]float64{
					+0.0000, +0.6234, +0.9749, +0.9009, // Thursday
					+0.4338, -0.2225, -0.7818, -1.0000, // Friday
					-0.7818, -0.2225, +0.4338, +0.9009, // Saturday
					+0.9749, +0.6234, +0.0000, -0.6234, // Sunday
					-0.9749, -0.9009, -0.4338, +0.2225, // Monday
					+0.7818, +1.0000, +0.7818, +0.2225, // Tuesday
					-0.4338, -0.9009, -0.9749, -0.6234, // Wednesday
				},
			).Set(
				feature.NewSeasonality("epoch_weekly", feature.FourierCompCos, 3),
				[]float64{
					+1.0000, +0.7818, +0.2225, -0.4338, // Thursday
					-0.9009, -0.9749, -0.6234, +0.0000, // Friday
					+0.6234, +0.9749, +0.9009, +0.4338, // Saturday
					-0.2225, -0.7818, -1.0000, -0.7818, // Sunday
					-0.2225, +0.4338, +0.9009, +0.9749, // Monday
					+0.6234, +0.0000, -0.6234, -0.9749, // Tuesday
					-0.9009, -0.4338, +0.2225, +0.7818, // Wednesday
				},
			),
			err: nil,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			tFeat, _ := generateTimeFeatures(tSeries, td.opt)
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

	chpntBias8hr := []float64{
		0, 0, 0, 0, // Thursday
		0, 0, 0, 0, // Friday
		0, 0, 0, 0, // Saturday
		0, 0, 0, 0, // Sunday
		0, 0, 0, 0, // Monday
		1, 1, 1, 1, // Tuesday
		1, 1, 1, 1, // Wednesday
	}
	tSeries := timedataset.GenerateT(4*7, 6*time.Hour, nowFunc)
	testData := map[string]struct {
		chpts           []Changepoint
		trainingEndTime time.Time
		enableGrowth    bool
		expected        *feature.Set
	}{
		"no changepoints": {
			chpts:           []Changepoint{},
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"changepoint after training end": {
			chpts: []Changepoint{
				{Name: "chpt1", T: endTime.Add(1 * time.Minute)},
			},
			trainingEndTime: endTime,
			expected:        feature.NewSet(),
		},
		"valid single changepoint": {
			chpts: []Changepoint{
				{Name: "chpt1", T: endTime.Add(-8 * 6 * time.Hour)},
			},
			trainingEndTime: endTime,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt1", feature.ChangepointCompBias),
				chpntBias8hr,
			),
		},
		"valid single changepoint with growth": {
			chpts: []Changepoint{
				{Name: "chpt_with_growth", T: endTime.Add(-8 * 6 * time.Hour)},
			},
			trainingEndTime: endTime,
			enableGrowth:    true,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt_with_growth", feature.ChangepointCompBias),
				chpntBias8hr,
			).Set(
				feature.NewChangepoint("chpt_with_growth", feature.ChangepointCompSlope),
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
			chpts: []Changepoint{
				{Name: "chpt_with_growth_and_future_training", T: endTime.Add(-12 * 6 * time.Hour)},
			},
			trainingEndTime: endTime.Add(8 * 6 * time.Hour),
			enableGrowth:    true,
			expected: feature.NewSet().Set(
				feature.NewChangepoint("chpt_with_growth_and_future_training", feature.ChangepointCompBias),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					1, 1, 1, 1, // Monday
					1, 1, 1, 1, // Tuesday
					1, 1, 1, 1, // Wednesday
				},
			).Set(
				feature.NewChangepoint("chpt_with_growth_and_future_training", feature.ChangepointCompSlope),
				[]float64{
					0, 0, 0, 0, // Thursday
					0, 0, 0, 0, // Friday
					0, 0, 0, 0, // Saturday
					0, 0, 0, 0, // Sunday
					0.00, 0.05, 0.10, 0.15, // Monday
					0.20, 0.25, 0.30, 0.35, // Tuesday
					0.40, 0.45, 0.50, 0.55, // Wednesday
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
