package forecast

import "github.com/aouyang1/go-forecaster/changepoint"

type Options struct {
	ChangepointOptions ChangepointOptions `json:"changepoint_options"`
	Regularization     float64            `json:"regularization"`
	DailyOrders        int                `json:"daily_orders"`
	WeeklyOrders       int                `json:"weekly_orders"`
}

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

type ChangepointOptions struct {
	Changepoints        []changepoint.Changepoint `json:"changepoints"`
	Auto                bool                      `json:"auto"`
	AutoNumChangepoints int                       `json:"auto_num_changepoints"`
}

func NewDefaultChangepointOptions() ChangepointOptions {
	return ChangepointOptions{
		Auto:                false,
		AutoNumChangepoints: DefaultAutoNumChangepoints,
		Changepoints:        nil,
	}
}
