package options

import (
	"bytes"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSeasonalityTablePrint(t *testing.T) {
	testData := map[string]struct {
		opt          *SeasonalityOptions
		prefix       string
		indent       string
		indentGrowth int
		expected     string
	}{
		"no configs": {
			opt: &SeasonalityOptions{},
			expected: `Seasonality: None
`,
		},
		"no configs with prefix and indent": {
			opt:          &SeasonalityOptions{},
			prefix:       "  ",
			indent:       "--",
			indentGrowth: 1,
			expected: `  --Seasonality: None
`,
		},
		"config with prefix and indent": {
			opt: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "s0", Period: 12 * time.Hour, Orders: 1},
				},
			},
			prefix:       "  ",
			indent:       "  ",
			indentGrowth: 1,
			expected: `    Seasonality:
       Name  Period Orders
         s0 12h0m0s      1
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

func TestRemoveDuplicates(t *testing.T) {
	testData := map[string]struct {
		opt      *SeasonalityOptions
		expected *SeasonalityOptions
	}{
		"no configs": {
			opt:      &SeasonalityOptions{},
			expected: &SeasonalityOptions{},
		},
		"period ordering": {
			opt: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "bar", Orders: 2, Period: 2 * time.Hour},
					{Name: "foo", Orders: 2, Period: 1 * time.Hour},
					{Name: "baz", Orders: 2, Period: 3 * time.Hour},
				},
			},
			expected: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "foo", Orders: 2, Period: 1 * time.Hour},
					{Name: "bar", Orders: 2, Period: 2 * time.Hour},
					{Name: "baz", Orders: 2, Period: 3 * time.Hour},
				},
			},
		},
		"orders ordering": {
			opt: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "bar", Orders: 2, Period: 2 * time.Hour},
					{Name: "foo", Orders: 1, Period: 2 * time.Hour},
					{Name: "baz", Orders: 3, Period: 2 * time.Hour},
				},
			},
			expected: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "baz", Orders: 3, Period: 2 * time.Hour},
				},
			},
		},
		"name ordering": {
			opt: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "bar", Orders: 1, Period: 1 * time.Hour},
					{Name: "foo", Orders: 1, Period: 1 * time.Hour},
					{Name: "baz", Orders: 1, Period: 1 * time.Hour},
				},
			},
			expected: &SeasonalityOptions{
				SeasonalityConfigs: []SeasonalityConfig{
					{Name: "bar", Orders: 1, Period: 1 * time.Hour},
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			td.opt.removeDuplicates()
			assert.Equal(t, td.expected, td.opt)
		})
	}
}
