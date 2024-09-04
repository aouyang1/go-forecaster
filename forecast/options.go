package forecast

import "github.com/aouyang1/go-forecast/changepoint"

type Options struct {
	Changepoints []changepoint.Changepoint
	DailyOrders  int
	WeeklyOrders int
}

func NewDefaultOptions() *Options {
	return &Options{
		Changepoints: nil,
		DailyOrders:  12,
		WeeklyOrders: 6,
	}
}
