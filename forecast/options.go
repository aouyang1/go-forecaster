package forecast

import (
	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/event"
)

// Options configures a forecast by specifying changepoints, seasonality order
// and an optional regularization parameter where higher values removes more features
// that contribute the least to the fit.
type Options struct {
	ChangepointOptions ChangepointOptions `json:"changepoint_options"`
	Regularization     float64            `json:"regularization"`
	DailyOrders        int                `json:"daily_orders"`
	WeeklyOrders       int                `json:"weekly_orders"`
	DSTOptions         DSTOptions         `json:"dst_options"`
	WeekendOptions     WeekendOptions     `json:"weekend_options"`
	EventOptions       EventOptions       `json:"event_options"`
}

// NewDefaultOptions returns a set of default forecast options
func NewDefaultOptions() *Options {
	return &Options{
		ChangepointOptions: NewDefaultChangepointOptions(),
		Regularization:     0.0,
		DailyOrders:        12,
		WeeklyOrders:       6,
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
	Changepoints        []changepoint.Changepoint `json:"changepoints"`
	EnableGrowth        bool                      `json:"enable_growth"`
	Auto                bool                      `json:"auto"`
	AutoNumChangepoints int                       `json:"auto_num_changepoints"`
}

// NewDefaultChangepointOptions generates a set of default changepoint options
func NewDefaultChangepointOptions() ChangepointOptions {
	return ChangepointOptions{
		Auto:                false,
		AutoNumChangepoints: DefaultAutoNumChangepoints,
		Changepoints:        nil,
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
	Enabled          bool   `json:"enabled"`
	TimezoneOverride string `json:"timezone_override"`
}

type EventOptions struct {
	Events []event.Event `json:"events"`
}
