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
			if i == 0 {
				filteredT = append(filteredT, t[i])
			}

			if math.IsNaN(y[i][j]) {
				lineData[i] = append(lineData[i], opts.LineData{Value: "-"})
			} else {
				lineData[i] = append(lineData[i], opts.LineData{Value: y[i][j]})
			}
		}
	}

	line.SetXAxis(filteredT)
	for i, series := range seriesName {
		line.AddSeries(series, lineData[i])
	}

	return line
}

// LineForecaster generates an echart line chart for a give fit result plotting the expected values
// along with the forecasted, upper, lower values.
func LineForecaster(trainingData *timedataset.TimeDataset, fitRes, forecastRes *Results) *charts.Line {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(
			opts.Title{
				Title: "Forecast Fit",
			},
		),
	)

	dataTime := make([]time.Time, 0, len(fitRes.T)+len(forecastRes.T))
	dataTime = append(dataTime, fitRes.T...)
	dataTime = append(dataTime, forecastRes.T...)

	lineDataActual := make([]opts.LineData, 0, len(trainingData.T))

	lineDataForecast := make([]opts.LineData, 0, len(fitRes.T)+len(forecastRes.T))
	lineDataUpper := make([]opts.LineData, 0, len(fitRes.T)+len(forecastRes.T))
	lineDataLower := make([]opts.LineData, 0, len(fitRes.T)+len(forecastRes.T))

	for i := 0; i < len(fitRes.T); i++ {
		lineDataActual = append(lineDataActual, opts.LineData{Value: trainingData.Y[i]})
		lineDataForecast = append(lineDataForecast, opts.LineData{Value: fitRes.Forecast[i]})
		lineDataUpper = append(lineDataUpper, opts.LineData{Value: fitRes.Upper[i]})
		lineDataLower = append(lineDataLower, opts.LineData{Value: fitRes.Lower[i]})
	}

	for i := 0; i < len(forecastRes.T); i++ {
		lineDataForecast = append(lineDataForecast, opts.LineData{Value: forecastRes.Forecast[i]})
		lineDataUpper = append(lineDataUpper, opts.LineData{Value: forecastRes.Upper[i]})
		lineDataLower = append(lineDataLower, opts.LineData{Value: forecastRes.Lower[i]})
	}

	line.SetXAxis(dataTime).
		AddSeries("Actual", lineDataActual).
		AddSeries("Forecast", lineDataForecast).
		AddSeries("Upper", lineDataUpper).
		AddSeries("Lower", lineDataLower)
	return line
}
