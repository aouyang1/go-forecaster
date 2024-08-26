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

	"github.com/aouyang1/go-forecast/forecast"
	"github.com/aouyang1/go-forecast/timedataset"
	"gonum.org/v1/gonum/floats"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/components"
	"github.com/go-echarts/go-echarts/v2/opts"
)

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

func ExampleForecaster() {
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
		y = append(y, 98.3+10.5*math.Sin(2.0*math.Pi/86400.0*float64(t[i].Unix()+2*60*60))+10.5*math.Cos(2.0*math.Pi*3.0/86400.0*float64(t[i].Unix()+2*60*60))+noise)
	}

	// add in anomalies
	anomalyRegion1 := y[len(y)/3 : len(y)/3+len(y)/20]
	floats.Scale(0, anomalyRegion1)
	floats.AddConst(2.7, anomalyRegion1)

	anomalyRegion2 := y[len(y)*2/3 : len(y)*2/3+len(y)/40]
	floats.AddConst(31.4, anomalyRegion2)

	opt := &Options{
		SeriesOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
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
	eq, err := f.seriesForecast.ModelEq()
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, eq)

	eq, err = f.residualForecast.ModelEq()
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, eq)

	res, err := f.Predict(td.T)
	if err != nil {
		panic(err)
	}
	page := components.NewPage()
	page.AddCharts(
		lineForecaster(td, res),
	)
	file, err := os.Create("examples/forecaster.html")
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
