package forecaster

import (
	"math"
	"time"

	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
)

// LineTSeries generates an echart multi-line chart for some arbitrary time/value combination. The input
// y is a slice of series that much have the same length as the input time slice.
func LineTSeries(title string, seriesName []string, t []time.Time, y [][]float64) *charts.Line {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(
			opts.Title{
				Title: title,
			},
		),
	)

	lineData := make([][]opts.LineData, len(y))

	filteredT := make([]time.Time, 0, len(t))
	for i := 0; i < len(y); i++ {
		lineData[i] = make([]opts.LineData, 0, len(y[i]))
		for j := 0; j < len(y[i]); j++ {
			if math.IsNaN(y[i][j]) {
				continue
			}
			if i == 0 {
				filteredT = append(filteredT, t[i])
			}
			lineData[i] = append(lineData[i], opts.LineData{Value: y[i][j]})
		}
	}

	line = line.SetXAxis(filteredT)
	for i, series := range seriesName {
		line = line.AddSeries(series, lineData[i])
	}

	return line
}

// LineForecaster generates an echart line chart for a give fit result plotting the expected values
// along with the forecasted, upper, lower values.
func LineForecaster(trainingData *timedataset.TimeDataset, res *Results) *charts.Line {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(
			opts.Title{
				Title: "Forecast Fit",
			},
		),
	)

	lineDataActual := make([]opts.LineData, 0, len(trainingData.T))
	lineDataForecast := make([]opts.LineData, 0, len(res.T))
	lineDataUpper := make([]opts.LineData, 0, len(res.T))
	lineDataLower := make([]opts.LineData, 0, len(res.T))

	for i := 0; i < len(res.T); i++ {
		lineDataActual = append(lineDataActual, opts.LineData{Value: trainingData.Y[i]})
		lineDataForecast = append(lineDataForecast, opts.LineData{Value: res.Forecast[i]})
		lineDataUpper = append(lineDataUpper, opts.LineData{Value: res.Upper[i]})
		lineDataLower = append(lineDataLower, opts.LineData{Value: res.Lower[i]})
	}

	line.SetXAxis(res.T).
		AddSeries("Actual", lineDataActual).
		AddSeries("Forecast", lineDataForecast).
		AddSeries("Upper", lineDataUpper).
		AddSeries("Lower", lineDataLower)
	return line
}
