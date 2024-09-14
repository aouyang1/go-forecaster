package forecast

import "github.com/aouyang1/go-forecaster/changepoint"

type Options struct {
	Changepoints []changepoint.Changepoint `json:"changepoints"`
	DailyOrders  int                       `json:"daily_orders"`
	WeeklyOrders int                       `json:"weekly_orders"`
}

func NewDefaultOptions() *Options {
	return &Options{
		Changepoints: nil,
		DailyOrders:  12,
		WeeklyOrders: 6,
	}
}
