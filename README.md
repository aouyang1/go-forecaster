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
  "time"

  "github.com/aouyang1/go-forecaster"
)

func MyForecast(t []time.Time, y []float64) ([]float64, error) {
  // initialize forecaster
  f, err := forecaster.New(nil)
  if err != nil {
    return nil, err
  }

  // fit the model
  if err := f.Fit(t, y); err != nil {
    return nil, err
  }

  // create future time slice with 10 samples after last training time at minute level
  // granularity
  horizon, err := f.MakeFuturePeriods(10, time.Minute)
  if err != nil {
    return nil, err
  }
  return f.Predict(horizon)
}	
```


