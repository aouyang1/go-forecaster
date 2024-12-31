package options

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	TZAmericaLosAngeles = "America/Los_Angeles"
	TZEuropeLondon      = "Europe/London"
)

func TestAdjustTime(t *testing.T) {
	londonTransitionTimes := []time.Time{
		time.Date(2024, 10, 26, 23, 0, 0, 0, time.UTC),
		time.Date(2024, 10, 27, 0, 0, 0, 0, time.UTC),
		time.Date(2024, 10, 27, 1, 0, 0, 0, time.UTC),
		time.Date(2024, 10, 27, 2, 0, 0, 0, time.UTC),
	}
	losAngelesTransitionTimes := []time.Time{
		time.Date(2024, 11, 3, 7, 0, 0, 0, time.UTC),
		time.Date(2024, 11, 3, 8, 0, 0, 0, time.UTC),
		time.Date(2024, 11, 3, 9, 0, 0, 0, time.UTC),
		time.Date(2024, 11, 3, 10, 0, 0, 0, time.UTC),
	}
	testData := map[string]struct {
		opt      DSTOptions
		t        []time.Time
		expected []time.Time
	}{
		"disabled": {
			opt: DSTOptions{},
			t: []time.Time{
				time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 1, 13, 0, 0, 0, time.UTC),
			},
			expected: []time.Time{
				time.Date(1970, 1, 1, 12, 0, 0, 0, time.UTC),
				time.Date(1970, 1, 1, 13, 0, 0, 0, time.UTC),
			},
		},
		"enabled no locations": {
			opt: DSTOptions{
				Enabled: true,
			},
			t: losAngelesTransitionTimes,
			expected: []time.Time{
				time.Date(2024, 11, 3, 7, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 8, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 9, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 10, 0, 0, 0, time.UTC),
			},
		},
		"enabled with one zone": {
			opt: DSTOptions{
				Enabled:           true,
				TimezoneLocations: []string{"America/Los_Angeles"},
			},
			t: losAngelesTransitionTimes,
			expected: []time.Time{
				time.Date(2024, 11, 3, 8, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 9, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 9, 0, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 10, 0, 0, 0, time.UTC),
			},
		},
		"enabled multiple": {
			opt: DSTOptions{
				Enabled:           true,
				TimezoneLocations: []string{"America/Los_Angeles", "Europe/London"},
			},
			t: append(londonTransitionTimes, losAngelesTransitionTimes...),
			expected: []time.Time{
				time.Date(2024, 10, 27, 0, 0, 0, 0, time.UTC),
				time.Date(2024, 10, 27, 1, 0, 0, 0, time.UTC),
				time.Date(2024, 10, 27, 1, 30, 0, 0, time.UTC), // London moves back 1hr
				time.Date(2024, 10, 27, 2, 30, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 7, 30, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 8, 30, 0, 0, time.UTC),
				time.Date(2024, 11, 3, 9, 0, 0, 0, time.UTC), // Los Angeles moves back 1hr
				time.Date(2024, 11, 3, 10, 0, 0, 0, time.UTC),
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.opt.AdjustTime(td.t)
			assert.Equal(t, td.expected, res)
		})
	}
}

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

func TestAdjustTimeSingle(t *testing.T) {
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
