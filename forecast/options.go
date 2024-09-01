package forecast

import "time"

type Options struct {
	Changepoints []time.Time
	DailyOrders  int
	WeeklyOrders int
}

func NewDefaultOptions() *Options {
	return &Options{
		DailyOrders:  12,
		WeeklyOrders: 6,
		Changepoints: nil,
	}
}
