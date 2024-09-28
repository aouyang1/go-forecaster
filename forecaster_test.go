package forecaster

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
	"time"

	"gonum.org/v1/gonum/mat"

	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/forecast"
	"gonum.org/v1/gonum/floats"
)

func generateExampleSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
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
			ChangepointOptions: forecast.ChangepointOptions{
				Changepoints: changepoints,
			},
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			ChangepointOptions: forecast.ChangepointOptions{
				Changepoints: changepoints,
			},
		},
		OutlierOptions: NewOutlierOptions(),
		ResidualWindow: 100,
		ResidualZscore: 4.0,
	}

	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m := f.Model()
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, string(out))

	if err := f.PlotFit("examples/forecaster.html"); err != nil {
		panic(err)
	}
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

func ExampleForecasterAutoChangepoint() {
	t, y := generateExampleSeries()

	opt := &Options{
		SeriesOptions: &forecast.Options{
			Regularization: 200.0,
			DailyOrders:    12,
			WeeklyOrders:   12,
			ChangepointOptions: forecast.ChangepointOptions{
				Auto:                true,
				AutoNumChangepoints: 100,
			},
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			ChangepointOptions: forecast.ChangepointOptions{
				Auto:         false,
				Changepoints: []changepoint.Changepoint{},
			},
		},
		OutlierOptions: NewOutlierOptions(),
		ResidualWindow: 100,
		ResidualZscore: 4.0,
	}
	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m := f.Model()
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, string(out))

	if err := f.PlotFit("examples/forecaster_auto_changepoint.html"); err != nil {
		panic(err)
	}
	// Output:
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
			ChangepointOptions: forecast.ChangepointOptions{
				Changepoints: changepoints,
			},
		},
		ResidualOptions: &forecast.Options{
			DailyOrders:  12,
			WeeklyOrders: 12,
			ChangepointOptions: forecast.ChangepointOptions{
				Changepoints: changepoints,
			},
		},
		OutlierOptions: NewOutlierOptions(),
		ResidualWindow: 100,
		ResidualZscore: 4.0,
	}
	f, err := New(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(t, y); err != nil {
		panic(err)
	}

	m := f.Model()
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Fprintln(os.Stderr, string(out))

	if err := f.PlotFit("examples/forecaster_with_trend.html"); err != nil {
		panic(err)
	}
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
