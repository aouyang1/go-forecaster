package forecast

type Options struct {
	DailyOrders  int
	WeeklyOrders int
}

func NewDefaultOptions() *Options {
	return &Options{
		DailyOrders:  12,
		WeeklyOrders: 6,
	}
}
