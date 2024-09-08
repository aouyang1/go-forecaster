package forecast

import "github.com/aouyang1/go-forecast/changepoint"

type Options struct {
	Changepoints []changepoint.Changepoint `json:"changepoints"`
	DailyOrders  int                       `json:"daily_orders"`
	WeeklyOrders int                       `json:"weekly_oroders"`
}

func NewDefaultOptions() *Options {
	return &Options{
		Changepoints: nil,
		DailyOrders:  12,
		WeeklyOrders: 6,
	}
}
