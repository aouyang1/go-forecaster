package forecaster

import (
	"math"
	"slices"
	"time"

	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/stats"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
)

// LineTSeries generates an echart multi-line chart for some arbitrary time/value combination. The input
// y is a slice of series that much have the same length as the input time slice.
func LineTSeries(title string, seriesName []string, t []time.Time, y [][]float64, forecastStartIdx int) *charts.Line {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(
			opts.Title{
				Title: title,
			},
		),
		charts.WithDataZoomOpts(
			opts.DataZoom{
				Type:       "slider",
				XAxisIndex: []int{0},
			},
		),
		charts.WithTooltipOpts(
			opts.Tooltip{
				Trigger: "axis",
			},
		),
	)

	lineData := make([][]opts.LineData, len(y))

	filteredT := make([]time.Time, 0, len(t))
	for i := range len(y) {
		lineData[i] = make([]opts.LineData, 0, len(y[i]))
		for j := 0; j < len(y[i]); j++ {
			if i == 0 {
				filteredT = append(filteredT, t[j])
			}

			lineData[i] = append(lineData[i], opts.LineData{Value: handleNaN(y[i][j])})
		}
	}

	markLineOpts := []charts.SeriesOpts{
		charts.WithMarkLineNameXAxisItemOpts(
			opts.MarkLineNameXAxisItem{
				XAxis: forecastStartIdx,
			},
		),
		charts.WithMarkLineStyleOpts(
			opts.MarkLineStyle{
				Symbol:    []string{"none", "none"},
				Label:     &opts.Label{Show: opts.Bool(false)},
				LineStyle: &opts.LineStyle{Color: "black"},
			},
		),
	}

	line.SetXAxis(filteredT)
	for i, series := range seriesName {
		if i == 0 {
			line.AddSeries(series, lineData[i], markLineOpts...)
			continue
		}
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
		charts.WithDataZoomOpts(
			opts.DataZoom{
				Type:       "slider",
				XAxisIndex: []int{0},
			},
		),
		charts.WithTooltipOpts(
			opts.Tooltip{
				Trigger: "axis",
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
		lineDataActual = append(lineDataActual, opts.LineData{Value: handleNaN(trainingData.Y[i])})
		lineDataForecast = append(lineDataForecast, opts.LineData{Value: handleNaN(fitRes.Forecast[i])})
		lineDataUpper = append(lineDataUpper, opts.LineData{Value: handleNaN(fitRes.Upper[i])})
		lineDataLower = append(lineDataLower, opts.LineData{Value: handleNaN(fitRes.Lower[i])})
	}

	for i := 0; i < len(forecastRes.T); i++ {
		lineDataForecast = append(lineDataForecast, opts.LineData{Value: handleNaN(forecastRes.Forecast[i])})
		lineDataUpper = append(lineDataUpper, opts.LineData{Value: handleNaN(forecastRes.Upper[i])})
		lineDataLower = append(lineDataLower, opts.LineData{Value: handleNaN(forecastRes.Lower[i])})
	}

	markLineOpts := []charts.SeriesOpts{
		charts.WithMarkLineNameXAxisItemOpts(
			opts.MarkLineNameXAxisItem{
				XAxis: len(fitRes.T),
			},
		),
		charts.WithMarkLineStyleOpts(
			opts.MarkLineStyle{
				Symbol:    []string{"none", "none"},
				Label:     &opts.Label{Show: opts.Bool(false)},
				LineStyle: &opts.LineStyle{Color: "black"},
			},
		),
	}
	line.SetXAxis(dataTime).
		AddSeries("Upper", lineDataUpper).
		AddSeries("Actual", lineDataActual).
		AddSeries("Forecast", lineDataForecast, markLineOpts...).
		AddSeries("Lower", lineDataLower)
	return line
}

// handleNaN converts nans to a string "-" to satisfy echarts requirement
func handleNaN(val float64) any {
	if math.IsNaN(val) {
		return "-"
	}
	return val
}

// removeOutlierEvents does an inplace setting of the training data to NaN if the time range
// is within or equal to the events' start/end times
func removeOutlierEvents(t []time.Time, y []float64, events []options.Event) {
	if len(events) == 0 {
		return
	}
	for i, ts := range t {
		if math.IsNaN(y[i]) {
			continue
		}
		for event := range slices.Values(events) {
			if math.IsNaN(y[i]) {
				// already removed from prior event due to overlapping events
				continue
			}

			if ts.After(event.Start) && ts.Before(event.End) {
				y[i] = math.NaN()
				continue
			}
			if ts.Equal(event.Start) || ts.Equal(event.End) {
				y[i] = math.NaN()
				continue
			}
		}
	}
}

// autoRemoveOutliers uses the Tukey method for detecting outliers and setting those data points
// to NaN. We detect outliers from residuals of the fit instead of the original data itself.
// We return number of outlier detected so that when none are found, the caller can early
// break out
func autoRemoveOutliers(t []time.Time, y, residual []float64, opts *OutlierOptions) int {
	if opts == nil || opts.NumPasses == 0 {
		return 0
	}

	outlierIdxs := stats.DetectOutliers(
		residual,
		opts.LowerPercentile,
		opts.UpperPercentile,
		opts.TukeyFactor,
	)
	outlierSet := make(map[int]struct{})
	for _, idx := range outlierIdxs {
		outlierSet[idx] = struct{}{}
	}

	// no more outliers detected with outlier options so break early
	if len(outlierIdxs) == 0 {
		return 0
	}

	for i := range len(t) {
		if _, exists := outlierSet[i]; exists {
			y[i] = math.NaN()
			continue
		}
	}
	return len(outlierIdxs)
}
