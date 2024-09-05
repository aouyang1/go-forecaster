# go-forecast
forecasting library

[![Go Report Card](https://goreportcard.com/badge/github.com/aouyang1/go-forecast)](https://goreportcard.com/report/github.com/aouyang1/go-forecast)
[![GoDoc](https://godoc.org/github.com/aouyang1/go-forecast?status.svg)](https://godoc.org/github.com/aouyang1/go-forecast)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Example 1
Contains a daily seasonal component with 2 anomalous behaviors along with 2 registered change points
![Forecast Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_example.png)

## Example 2
Contains a daily seasonal component with a trend change point which resets
![Forecast With Trend Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_with_trend_example.png)

## Example Use
```
import (
  "fmt"

  forecaster "github.com/aouyang1/go-forecast"
  "github.com/aouyang1/go-forecast/timedataset"
)

func MyForecast(t []time.Time, y []float64) (*forecaster.Results, error) {
	td, err := timedataset.NewUnivariateDataset(t, y)
	if err != nil {
		return nil, err
	}
	f, err := forecaster.New(opt)
	if err != nil {
		return nil, err
	}
	if err := f.Fit(td); err != nil {
		return nil, err
	}
	eq, err := f.seriesForecast.ModelEq()
	if err != nil {
	  return nil, err
	}
	fmt.Fprintln(os.Stderr, eq)

	return f.Predict(td.T)
}	
```


