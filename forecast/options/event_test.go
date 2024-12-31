package options

import (
	"bytes"
	"testing"
	"time"

	"github.com/rickar/cal/v2"
	"github.com/rickar/cal/v2/us"
	"github.com/stretchr/testify/assert"
)

func TestHoliday(t *testing.T) {
	testData := map[string]struct {
		hol       *cal.Holiday
		start     time.Time
		end       time.Time
		durBefore time.Duration
		durAfter  time.Duration
		expected  []Event
	}{
		"no coverage": {
			hol:       us.ChristmasDay,
			start:     time.Date(2024, 12, 8, 1, 0, 0, 0, time.UTC),
			end:       time.Date(2024, 12, 12, 1, 0, 0, 0, time.UTC),
			durBefore: 0,
			durAfter:  0,
			expected:  []Event{},
		},
		"simple": {
			hol:       us.ChristmasDay,
			start:     time.Date(2024, 12, 8, 1, 0, 0, 0, time.UTC),
			end:       time.Date(2026, 12, 8, 1, 0, 0, 0, time.UTC),
			durBefore: 0,
			durAfter:  0,
			expected: []Event{
				{
					"Christmas_Day_2024",
					time.Date(2024, 12, 25, 0, 0, 0, 0, time.UTC),
					time.Date(2024, 12, 26, 0, 0, 0, 0, time.UTC),
				},
				{
					"Christmas_Day_2025",
					time.Date(2025, 12, 25, 0, 0, 0, 0, time.UTC),
					time.Date(2025, 12, 26, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		"non utc tz": {
			hol:       us.ChristmasDay,
			start:     time.Date(2024, 12, 8, 1, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
			end:       time.Date(2026, 12, 8, 1, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
			durBefore: 0,
			durAfter:  0,
			expected: []Event{
				{
					"Christmas_Day_2024",
					time.Date(2024, 12, 25, 0, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
					time.Date(2024, 12, 26, 0, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
				},
				{
					"Christmas_Day_2025",
					time.Date(2025, 12, 25, 0, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
					time.Date(2025, 12, 26, 0, 0, 0, 0, time.FixedZone("UTC-8", -8*60*60)),
				},
			},
		},
		"with buffer": {
			hol:       us.ChristmasDay,
			start:     time.Date(2024, 12, 8, 1, 0, 0, 0, time.UTC),
			end:       time.Date(2026, 12, 8, 1, 0, 0, 0, time.UTC),
			durBefore: time.Duration(24 * time.Hour),
			durAfter:  time.Duration(2 * 24 * time.Hour),
			expected: []Event{
				{
					"Christmas_Day_2024",
					time.Date(2024, 12, 24, 0, 0, 0, 0, time.UTC),
					time.Date(2024, 12, 28, 0, 0, 0, 0, time.UTC),
				},
				{
					"Christmas_Day_2025",
					time.Date(2025, 12, 24, 0, 0, 0, 0, time.UTC),
					time.Date(2025, 12, 28, 0, 0, 0, 0, time.UTC),
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := Holiday(td.hol, td.start, td.end, td.durBefore, td.durAfter)
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestValid(t *testing.T) {
	testData := map[string]struct {
		name  string
		start time.Time
		end   time.Time
		err   error
	}{
		"unset start time": {
			end:  time.Now(),
			name: "blargh",
			err:  ErrUnsetTime,
		},
		"unset end time": {
			start: time.Now(),
			name:  "blargh",
			err:   ErrUnsetTime,
		},
		"start after end": {
			start: time.Now().Add(time.Hour),
			end:   time.Now(),
			name:  "blargh",
			err:   ErrStartAfterEnd,
		},
		"no name": {
			start: time.Now().Add(-time.Hour),
			end:   time.Now(),
			err:   ErrNoEventName,
		},
		"valid": {
			start: time.Now().Add(-time.Hour),
			end:   time.Now(),
			name:  "blargh",
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			e := NewEvent(td.name, td.start, td.end)
			err := e.Valid()
			if td.err != nil {
				assert.EqualError(t, err, td.err.Error())
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestEventTablePrint(t *testing.T) {
	testData := map[string]struct {
		opt          *EventOptions
		prefix       string
		indent       string
		indentGrowth int
		expected     string
	}{
		"no configs": {
			opt: &EventOptions{},
			expected: `Events: None
`,
		},
		"no configs with prefix and indent": {
			opt:          &EventOptions{},
			prefix:       "  ",
			indent:       "--",
			indentGrowth: 1,
			expected: `  --Events: None
`,
		},
		"config with prefix and indent": {
			opt: &EventOptions{
				Events: []Event{
					{Name: "e0", Start: time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC), End: time.Date(1970, 1, 2, 0, 0, 0, 0, time.UTC)},
				},
			},
			prefix:       "  ",
			indent:       "  ",
			indentGrowth: 1,
			expected: `    Events:
       Name                         Start                           End
         e0 1970-01-01 00:00:00 +0000 UTC 1970-01-02 00:00:00 +0000 UTC
`,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			td.opt.TablePrint(&buf, td.prefix, td.indent, td.indentGrowth)
			assert.Equal(t, td.expected, buf.String())
		})
	}
}
