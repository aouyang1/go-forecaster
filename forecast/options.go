package forecast

import "github.com/aouyang1/go-forecaster/changepoint"

// Options configures a forecast by specifying changepoints, seasonality order
// and an optional regularization parameter where higher values removes more features
// that contribute the least to the fit.
type Options struct {
	ChangepointOptions ChangepointOptions `json:"changepoint_options"`
	Regularization     float64            `json:"regularization"`
	DailyOrders        int                `json:"daily_orders"`
	WeeklyOrders       int                `json:"weekly_orders"`
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
