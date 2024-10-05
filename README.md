# go-forecaster
forecasting library

[![Go Report Card](https://goreportcard.com/badge/github.com/aouyang1/go-forecast)](https://goreportcard.com/report/github.com/aouyang1/go-forecast)
[![GoDoc](https://pkg.go.dev/badge/github.com/aouyang1/go-forecaster.svg)](https://pkg.go.dev/github.com/aouyang1/go-forecaster)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Example 1
Contains a daily seasonal component with 2 anomalous behaviors along with 2 registered change points
![Forecast Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_example.png)

## Example 2
Contains a daily seasonal component with a trend change point which resets
![Forecast With Trend Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_with_trend_example.png)

## Example 3
Auto-changepoint detection and fit
![Forecast With Auto-Changepoint Detection Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_with_auto_changepoint_example.png)


## Example Use
```
import (
  "fmt"

  "github.com/aouyang1/go-forecaster"
  "github.com/aouyang1/go-forecaster/timedataset"
)

func MyForecast(t []time.Time, y []float64) (*forecaster.Results, error) {
	f, err := forecaster.New(nil)
	if err != nil {
		return nil, err
	}
	if err := f.Fit(t, y); err != nil {
		return nil, err
	}
	eq, err := f.SeriesModelEq()
	if err != nil {
	  return nil, err
	}
	fmt.Fprintln(os.Stderr, eq)

	return f.Predict(td.T)
}	
```


