# go-forecast
forecasting library

![Forecast Example](https://github.com/aouyang1/go-forecast/tree/main/examples/forecast_example.png)

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
	
```


