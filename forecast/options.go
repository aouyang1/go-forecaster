package forecast

import (
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"gonum.org/v1/gonum/dsp/window"
)

const (
	WindowBartlettHann    = "bartlett_hann"
	WindowBlackman        = "blackman"
	WindowBlackmanHarris  = "blackman_harris"
	WindowBlackmanNuttall = "blackman_nuttall"
	WindowFlatTop         = "flat_top"
	WindowHamming         = "hamming"
	WindowHann            = "hann"
	WindowLanczos         = "lanczos"
	WindowNuttall         = "nuttall"
	WindowRectangular     = "rectangular"
	WindowSine            = "sine"
	WindowTriangular      = "triangular"
)

func WindowFunc(name string) func(seq []float64) []float64 {
	var winFunc func(seq []float64) []float64
	switch name {
	case WindowBartlettHann:
		winFunc = window.BartlettHann
	case WindowBlackman:
		winFunc = window.Blackman
	case WindowBlackmanHarris:
		winFunc = window.BlackmanHarris
	case WindowBlackmanNuttall:
		winFunc = window.BlackmanNuttall
	case WindowFlatTop:
		winFunc = window.FlatTop
	case WindowHamming:
		winFunc = window.Hamming
	case WindowHann:
		winFunc = window.Hann
	case WindowLanczos:
		winFunc = window.Lanczos
	case WindowNuttall:
		winFunc = window.Nuttall
	case WindowRectangular:
		winFunc = window.Rectangular
	case WindowSine:
		winFunc = window.Sine
	case WindowTriangular:
		winFunc = window.Triangular
	default:
		winFunc = window.Rectangular
	}
	return winFunc
}

// Options configures a forecast by specifying changepoints, seasonality order
// and an optional regularization parameter where higher values removes more features
// that contribute the least to the fit.
type Options struct {
	ChangepointOptions ChangepointOptions `json:"changepoint_options"`

	// Lasso related options
	Regularization  []float64 `json:"regularization"`
	Iterations      int       `json:"iterations"`
	Tolerance       float64   `json:"tolerance"`
	Parallelization int       `json:"parallelization"`

	SeasonalityOptions SeasonalityOptions `json:"seasonality_options"`

	DSTOptions     DSTOptions     `json:"dst_options"`
	WeekendOptions WeekendOptions `json:"weekend_options"`
	EventOptions   EventOptions   `json:"event_options"`
	MaskWindow     string         `json:"mask_window"`
}

// NewDefaultOptions returns a set of default forecast options
func NewDefaultOptions() *Options {
	return &Options{
		ChangepointOptions: NewDefaultChangepointOptions(),
		Regularization:     []float64{0.0},
		SeasonalityOptions: NewDefaultSeasonalityOptions(),
	}
}

const (
	DefaultAutoNumChangepoints int = 100
)

// ChangepointOptions configures the changepoint fit to either use auto-detection
// by evenly placing N changepoints in the training window. Running with auto-detection
// will generally require increasing the regularization parameter to remove changepoints
// that can cause overfitting. In general it is best to specify known changepoints which
// results in much faster training times and model size.
type ChangepointOptions struct {
	Changepoints        []Changepoint `json:"changepoints"`
	EnableGrowth        bool          `json:"enable_growth"`
	Auto                bool          `json:"auto"`
	AutoNumChangepoints int           `json:"auto_num_changepoints"`
}

func (c ChangepointOptions) tablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(c.Changepoints) > 0 {
		noCfg = ""
		fmt.Fprintf(tbl, "%s%sName\tDatetime\t\n", prefix, indentExpand(indent, indentGrowth+1))
	}
	fmt.Fprintf(w, "%s%sChangepoints:%s\n", prefix, indentExpand(indent, indentGrowth), noCfg)
	for _, chpt := range c.Changepoints {
		fmt.Fprintf(tbl, "%s%s%s\t%s\t\n",
			prefix, indentExpand(indent, indentGrowth+1),
			chpt.Name, chpt.T)
	}
	return tbl.Flush()
}

// NewDefaultChangepointOptions generates a set of default changepoint options
func NewDefaultChangepointOptions() ChangepointOptions {
	return ChangepointOptions{
		Auto:                false,
		AutoNumChangepoints: DefaultAutoNumChangepoints,
		Changepoints:        nil,
	}
}

// Seasonality options configures the number of seasonality components to fit for.
type SeasonalityOptions struct {
	SeasonalityConfigs []SeasonalityConfig `json:"seasonality_configs"`
}

func (s SeasonalityOptions) tablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(s.SeasonalityConfigs) > 0 {
		noCfg = ""
		fmt.Fprintf(tbl, "%s%sName\tPeriod\tOrders\t\n", prefix, indentExpand(indent, indentGrowth+1))
	}
	fmt.Fprintf(w, "%s%sSeasonality:%s\n", prefix, indentExpand(indent, indentGrowth), noCfg)
	for _, seasCfg := range s.SeasonalityConfigs {
		fmt.Fprintf(tbl, "%s%s%s\t%s\t%d\t\n",
			prefix, indentExpand(indent, indentGrowth+1),
			seasCfg.Name, seasCfg.Period, seasCfg.Orders)
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

// SeasonalityConfig represents a single seasonality configuration to model. This will generate
// Fourier series of the specified period and number of orders. E.g. a period of 24*time.Hour
// with 3 orders will create 6 Fourier series of order 1, 2, 3 and for the sine/cosine components
// where order 1 will have a period of 1 day and order 2 will have a period of 12 hours.
type SeasonalityConfig struct {
	Name   string        `json:"name"`
	Orders int           `json:"orders"`
	Period time.Duration `json:"period"`
}

// NewDailySeasonalityConfig creates a daily seasonality config given a specified number of orders
func NewDailySeasonalityConfig(orders int) SeasonalityConfig {
	if orders < 0 {
		orders = 0
	}

	return SeasonalityConfig{
		Name:   LabelSeasDaily,
		Orders: orders,
		Period: 24 * time.Hour,
	}
}

// NewWeeklySeasonalityConfig creates a weekly seasonality config given a specified number of orders
func NewWeeklySeasonalityConfig(orders int) SeasonalityConfig {
	if orders < 0 {
		orders = 0
	}

	return SeasonalityConfig{
		Name:   LabelSeasWeekly,
		Orders: orders,
		Period: 7 * 24 * time.Hour,
	}
}

// DSTOptions lets us adjust the time to account for Daylight Saving Time behavior changes
// by timezone. In the presence of multiple timezones this will average out the effect evenly
// across the input timezones. e.g America/Los_Angeles + Europe/London will shift the time by 30min
// 2024-03-10 (America) to 2024-03-31 (Europe) and then by 60min on or after 2024-03-31.
type DSTOptions struct {
	Enabled           bool     `json:"enabled"`
	TimezoneLocations []string `json:"timezone_locations"`
}

// WeekendOptions lets us model weekends separately from weekdays.
type WeekendOptions struct {
	Enabled          bool          `json:"enabled"`
	TimezoneOverride string        `json:"timezone_override"`
	DurBefore        time.Duration `json:"duration_before"`
	DurAfter         time.Duration `json:"duration_after"`
}

type EventOptions struct {
	Events []Event `json:"events"`
}

func (e EventOptions) tablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(e.Events) > 0 {
		noCfg = ""
		fmt.Fprintf(tbl, "%s%sName\tStart\tEnd\t\n", prefix, indentExpand(indent, indentGrowth+1))
	}
	fmt.Fprintf(w, "%s%sEvents:%s\n", prefix, indentExpand(indent, indentGrowth), noCfg)
	for _, ev := range e.Events {
		fmt.Fprintf(tbl, "%s%s%s\t%s\t%s\t\n",
			prefix, indentExpand(indent, indentGrowth+1),
			ev.Name, ev.Start, ev.End)
	}
	return tbl.Flush()
}
