package forecast

type Components struct {
	Trend       []float64 `json:"trend"`
	Seasonality []float64 `json:"seasonality"`
}