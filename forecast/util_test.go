package forecast

import (
	"testing"
	"time"

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
		return time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC)
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
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			features := generateTimeFeatures(td.t, td.opt)
			assert.Equal(t, td.expected, features)
		})
	}
}
