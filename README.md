# go-forecaster

[![Go Report Card](https://goreportcard.com/badge/github.com/aouyang1/go-forecast)](https://goreportcard.com/report/github.com/aouyang1/go-forecast)
[![codecov](https://codecov.io/gh/aouyang1/go-forecaster/graph/badge.svg?token=WjJ4sFBUrz)](https://codecov.io/gh/aouyang1/go-forecaster)
[![GoDoc](https://pkg.go.dev/badge/github.com/aouyang1/go-forecaster.svg)](https://pkg.go.dev/github.com/aouyang1/go-forecaster)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`go-forecaster` is a Go library designed for time-series forecasting. It enables users to model and predict data with strong seasonal components, events, holidays, change points, and trends. The library offers functionalities for fitting forecast models, making future predictions, and visualizing results using Apache ECharts.​

## Features
- **Model Fitting**: Fit time-series data to capture trends, seasonal patterns, and events.
- **Prediction**: Generate forecasts for future time points based on fitted models.
- **Visualization**: Create interactive line charts to visualize actual data alongside forecasts and confidence intervals.
- **Changepoint Detection**: Identify points where the time-series data exhibits abrupt changes in trend or seasonality.
- **Event Components**: Incorporate and analyze the impact of specific events on the time-series data.

## Examples

Contains a daily seasonal component with 2 anomalous behaviors along with 2 registered change points
![Forecast Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_example.png)

Contains a daily seasonal component with a trend change point which resets
![Forecast With Trend Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_with_trend_example.png)

Auto-changepoint detection and fit
![Forecast With Auto-Changepoint Detection Example](https://github.com/aouyang1/go-forecast/blob/main/examples/forecast_with_auto_changepoint_example.png)

## Installation
To install `go-forecaster`, use `go get`:

```bash
go get github.com/aouyang1/go-forecaster
```

## Usage
Here's a basic example demonstrating how to use go-forecaster:
```go
package main

import (
    "fmt"
    "time"

    "github.com/aouyang1/go-forecaster"
)

func main() {
    // Sample time-series data
    times := []time.Time{...} // Your time data here
    values := []float64{...}  // Corresponding values

    // Initialize the forecaster with default options
    f, err := forecaster.New(nil)
    if err != nil {
        fmt.Println("Error initializing forecaster:", err)
        return
    }

    // Fit the model to the data
    err = f.Fit(times, values)
    if err != nil {
        fmt.Println("Error fitting model:", err)
        return
    }

    // Generate future time points for prediction
    futureTimes, err := f.MakeFuturePeriods(10, 24*time.Hour)
    if err != nil {
        fmt.Println("Error generating future periods:", err)
        return
    }

    // Predict future values
    results, err := f.Predict(futureTimes)
    if err != nil {
        fmt.Println("Error making predictions:", err)
        return
    }

    // Output the predictions
    for i, t := range futureTimes {
        fmt.Printf("Date: %s, Forecast: %.2f\n", t.Format("2006-01-02"), results.Forecast[i])
    }
}
```
This example initializes a forecaster, fits it to sample data, and predicts future values. For more detailed examples, refer to the `examples` directory in the repository.

## Visualization
`go-forecaster` integrates with Apache ECharts to provide interactive visualizations of your forecasts. After fitting a model, you can generate an HTML visualization:

```go
// Assuming 'f' is your fitted Forecaster instance
err := f.PlotFit(outputWriter, nil)
if err != nil {
    fmt.Println("Error generating plot:", err)
}
```

This will create an HTML file displaying the original data, the forecast, and confidence intervals.

## Documentation
Comprehensive documentation is available on [pkg.go.dev](https://pkg.go.dev/github.com/aouyang1/go-forecaster), detailing all available functions and types.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! If you encounter issues or have suggestions for improvements, please open an issue or submit a pull request.
