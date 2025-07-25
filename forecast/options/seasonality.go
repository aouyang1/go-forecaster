package options

import (
	"fmt"
	"io"
	"slices"
	"sort"
	"text/tabwriter"
	"time"

	"github.com/aouyang1/go-forecaster/forecast/util"
)

// SeasonalityOptions configures the number of seasonality components to fit for.
type SeasonalityOptions struct {
	SeasonalityConfigs []SeasonalityConfig `json:"seasonality_configs"`
}

func (s SeasonalityOptions) TablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(s.SeasonalityConfigs) > 0 {
		noCfg = ""
		if _, err := fmt.Fprintf(tbl, "%s%sName\tPeriod\tOrders\t\n", prefix, util.IndentExpand(indent, indentGrowth+1)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s%sSeasonality:%s\n", prefix, util.IndentExpand(indent, indentGrowth), noCfg); err != nil {
		return err
	}
	for _, seasCfg := range s.SeasonalityConfigs {
		if _, err := fmt.Fprintf(tbl, "%s%s%s\t%s\t%d\t\n",
			prefix, util.IndentExpand(indent, indentGrowth+1),
			seasCfg.Name, seasCfg.Period, seasCfg.Orders); err != nil {
			return err
		}
	}
	return tbl.Flush()
}

// NewDefaultSeasonalityOptions generates a default seasonality config with weekly and daily
// seasonal components
func NewDefaultSeasonalityOptions() SeasonalityOptions {
	return SeasonalityOptions{
		SeasonalityConfigs: []SeasonalityConfig{
			NewDailySeasonalityConfig(12),
			NewWeeklySeasonalityConfig(6),
		},
	}
}

func (s *SeasonalityOptions) removeDuplicates() {
	// sort seasonality configs so we can find duplicate periods and remove them
	optSeasConfigs := s.SeasonalityConfigs
	sort.Slice(optSeasConfigs, func(i, j int) bool {
		if optSeasConfigs[i].Period < optSeasConfigs[j].Period {
			return true
		}
		if optSeasConfigs[i].Period > optSeasConfigs[j].Period {
			return false
		}
		if optSeasConfigs[i].Orders > optSeasConfigs[j].Orders {
			return true
		}
		if optSeasConfigs[i].Orders < optSeasConfigs[j].Orders {
			return false
		}
		return optSeasConfigs[i].Name < optSeasConfigs[j].Name
	})
	validIdx := make([]int, 0, len(optSeasConfigs))
	var lastValidPeriod time.Duration
	for i, seasCfg := range optSeasConfigs {
		if seasCfg.Period > 0 && seasCfg.Period > lastValidPeriod && seasCfg.Name != "" && seasCfg.Orders > 0 {
			validIdx = append(validIdx, i)
			lastValidPeriod = seasCfg.Period
		}
	}

	if len(validIdx) != len(optSeasConfigs) {
		validatedSeasConfigs := make([]SeasonalityConfig, 0, len(validIdx))
		for _, i := range validIdx {
			validatedSeasConfigs = append(validatedSeasConfigs, optSeasConfigs[i])
		}
		optSeasConfigs = validatedSeasConfigs
	}
	s.SeasonalityConfigs = optSeasConfigs
}

// colinearConfigOrders returns a mapping of seasonality configs to a slice or fourier orders that
// can be pruned
func (s *SeasonalityOptions) colinearConfigOrders() map[SeasonalityConfig][]int {
	periods := make(map[float64]struct{})
	colinearCfgOrders := make(map[SeasonalityConfig][]int)
	for _, seasCfg := range s.SeasonalityConfigs {
		for i := range seasCfg.Orders {
			period := float64(seasCfg.Period) / float64(i+1)
			if _, exists := periods[period]; exists {
				// store colinear period
				colinearCfgOrders[seasCfg] = append(colinearCfgOrders[seasCfg], i+1)
				continue
			}
			periods[period] = struct{}{}
		}
	}
	return colinearCfgOrders
}

// SeasonalityConfig represents a single seasonality configuration to model. This will generate
// Fourier series of the specified period and number of orders. E.g. a period of 24*time.Hour
// with 3 orders will create 6 Fourier series of order 1, 2, 3 and for the sine/cosine components
// where order 1 will have a period of 1 day and order 2 will have a period of 12 hours.
type SeasonalityConfig struct {
	Name   string        `json:"name"`
	Orders int           `json:"orders"`
	Period time.Duration `json:"period"`
}

// NewSeasonalityConfig creates a new seasonality config given a name, period and orders
func NewSeasonalityConfig(name string, period time.Duration, orders int) SeasonalityConfig {
	if orders < 0 {
		orders = 0
	}

	return SeasonalityConfig{
		Name:   name,
		Orders: orders,
		Period: period,
	}
}

// NewDailySeasonalityConfig creates a daily seasonality config given a specified number of orders
func NewDailySeasonalityConfig(orders int) SeasonalityConfig {
	return NewSeasonalityConfig(LabelSeasDaily, 24*time.Hour, orders)
}

// NewWeeklySeasonalityConfig creates a weekly seasonality config given a specified number of orders
func NewWeeklySeasonalityConfig(orders int) SeasonalityConfig {
	return NewSeasonalityConfig(LabelSeasWeekly, 7*24*time.Hour, orders)
}

// filterOutColinearOrders creates a slice or orders from one removing any previously declared
// colinear seasonal orders
func (s SeasonalityConfig) filterOutColinearOrders(colinearCfgOrders map[SeasonalityConfig][]int) []int {
	var orders []int
	for i := range s.Orders {
		colinearOrders, colinearCfgExists := colinearCfgOrders[s]
		if colinearCfgExists && slices.Contains(colinearOrders, i+1) {
			continue
		}
		orders = append(orders, i+1)
	}
	return orders
}
