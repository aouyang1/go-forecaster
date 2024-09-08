package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/aouyang1/go-forecast/changepoint"
	"github.com/aouyang1/go-forecast/forecast"
	"github.com/aouyang1/go-forecast/timedataset"
	"gonum.org/v1/gonum/floats"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
)

func lineTSeries(title string, seriesName []string, t []time.Time, y [][]float64) *charts.Line {
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

func lineForecaster(trainingData *timedataset.TimeDataset, res *Results) *charts.Line {
	line := charts.NewLine()
	line.SetGlobalOptions(
		charts.WithTitleOpts(
			opts.Title{
				Title: "Forecast example",
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

func generateExampleSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		t = append(t, ct.Add(-time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		noise := rand.NormFloat64() * (3.2 + 3.2*math.Sin(2.0*math.Pi*5.0/86400.0*float64(t[i].Unix())))
		bias := 98.3
		daily1 := 10.5 * math.Sin(2.0*math.Pi/86400.0*float64(t[i].Unix()+2*60*60))
		daily2 := 10.5 * math.Cos(2.0*math.Pi*3.0/86400.0*float64(t[i].Unix()+2*60*60))

		jump := 0.0
		if i > minutes/2 {
			jump = 10.0
		}
		if i > minutes*17/20 {
			jump = -60.0
		}
		y = append(y, bias+daily1+daily2+noise+jump)
	}

	// add in anomalies
	anomalyRegion1 := y[len(y)/3 : len(y)/3+len(y)/20]
	floats.Scale(0, anomalyRegion1)
	floats.AddConst(2.7, anomalyRegion1)

	anomalyRegion2 := y[len(y)*2/3 : len(y)*2/3+len(y)/40]
	floats.AddConst(61.4, anomalyRegion2)

	return t, y
}

func ExampleForecaster() {
	t, y := generateExampleSeries()

	changepoints := []changepoint.Changepoint{
		changepoint.New("anomaly1", t[len(t)/2]),
		changepoint.New("anomaly2", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			Changepoints: changepoints,
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			Changepoints: changepoints,
		},
		OutlierOptions: NewOutlierOptions(),
		ResidualWindow: 100,
		ResidualZscore: 4.0,
	}
	td, err := timedataset.NewUnivariateDataset(t, y)
	if err != nil {
		panic(err)
	}
	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(td); err != nil {
		panic(err)
	}

	intercept := f.SeriesIntercept()
	coef, err := f.SeriesCoefficients()
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "intercept: %.5f\n", intercept)
	for _, label := range f.residualForecast.FeatureLabels() {
		fmt.Fprintf(os.Stderr, "%s: %.5f\n", label, coef[label.String()])
	}

	intercept = f.ResidualIntercept()
	coef, err = f.ResidualCoefficients()
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "intercept: %.5f\n", intercept)
	for _, label := range f.residualForecast.FeatureLabels() {
		fmt.Fprintf(os.Stderr, "%s: %.5f\n", label, coef[label.String()])
	}

	res, err := f.Predict(td.T)
	if err != nil {
		panic(err)
	}
	page := components.NewPage()
	page.AddCharts(
		lineForecaster(td, res),
		lineTSeries(
			"Forecast Components",
			[]string{"Trend", "Seasonality"},
			td.T,
			[][]float64{
				f.TrendComponent(),
				f.SeasonalityComponent(),
			},
		),
		lineTSeries(
			"Forecast Residual",
			[]string{"Residual"},
			td.T,
			[][]float64{f.Residuals()},
		),
	)
	file, err := os.Create("examples/forecaster.html")
	if err != nil {
		panic(err)
	}
	page.Render(io.MultiWriter(file))
	// Output:
}

func generateExampleSeriesWithTrend() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(24*4) * time.Hour)
	for i := 0; i < minutes; i++ {
		t = append(t, ct.Add(time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		noise := rand.NormFloat64() * (3.2 + 3.2*math.Sin(2.0*math.Pi*5.0/86400.0*float64(t[i].Unix())))
		bias := 98.3
		daily1 := 10.5 * math.Sin(2.0*math.Pi/86400.0*float64(t[i].Unix()+2*60*60))
		daily2 := 10.5 * math.Cos(2.0*math.Pi*3.0/86400.0*float64(t[i].Unix()+2*60*60))

		jump := 0.0
		if i > minutes/2 && i < minutes*17/20 {
			jump = 40.0 / float64(minutes*17/20-minutes/2) * float64(i-minutes/2)
		}
		y = append(y, bias+daily1+daily2+noise+jump)
	}

	return t, y
}

func ExampleForecasterWithTrend() {
	t, y := generateExampleSeriesWithTrend()

	changepoints := []changepoint.Changepoint{
		changepoint.New("trendstart", t[len(t)/2]),
		changepoint.New("rebaseline", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			Changepoints: changepoints,
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			Changepoints: changepoints,
		},
		OutlierOptions: NewOutlierOptions(),
		ResidualWindow: 100,
		ResidualZscore: 4.0,
	}
	td, err := timedataset.NewUnivariateDataset(t, y)
	if err != nil {
		panic(err)
	}
	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(td); err != nil {
		panic(err)
	}

	intercept := f.SeriesIntercept()
	coef, err := f.SeriesCoefficients()
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "intercept: %.5f\n", intercept)
	for _, label := range f.residualForecast.FeatureLabels() {
		fmt.Fprintf(os.Stderr, "%s: %.5f\n", label, coef[label.String()])
	}

	intercept = f.ResidualIntercept()
	coef, err = f.ResidualCoefficients()
	if err != nil {
		panic(err)
	}
	fmt.Fprintf(os.Stderr, "intercept: %.5f\n", intercept)
	for _, label := range f.residualForecast.FeatureLabels() {
		fmt.Fprintf(os.Stderr, "%s: %.5f\n", label, coef[label.String()])
	}

	res, err := f.Predict(td.T)
	if err != nil {
		panic(err)
	}
	page := components.NewPage()
	page.AddCharts(
		lineForecaster(td, res),
		lineTSeries(
			"Forecast Components",
			[]string{"Trend", "Seasonality"},
			td.T,
			[][]float64{
				f.TrendComponent(),
				f.SeasonalityComponent(),
			},
		),
		lineTSeries(
			"Forecast Residual",
			[]string{"Residual"},
			td.T,
			[][]float64{f.seriesForecast.Residuals()},
		),
	)
	file, err := os.Create("examples/forecaster_with_trend.html")
	if err != nil {
		panic(err)
	}
	page.Render(io.MultiWriter(file))
	// Output:
}

func TestMatrixMulWithNaN(t *testing.T) {
	// Initialize two matrices, a and b.
	a := mat.NewDense(1, 2, []float64{
		4, 3,
	})
	b := mat.NewDense(2, 4, []float64{
		4, 0, 0, math.NaN(),
		0, 0, 4, math.NaN(),
	})

	// Take the matrix product of a and b and place the result in c.
	var c mat.Dense
	c.Mul(a, b)

	// Print the result using the formatter.
	fc := mat.Formatted(&c, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v", fc)
	// Output: c = [16  0  12  NaN]
}
