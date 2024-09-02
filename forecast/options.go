package forecast

import "time"

type Options struct {
	Changepoints []time.Time
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
