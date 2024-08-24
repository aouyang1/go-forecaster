package main

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestFit(t *testing.T) {
	// create a daily sine wave at minutely with one week
	minutes := 24 * 60
	bias := 1.2
	amp := 4.3
	phase := int64(3 * 60 * 60) // 3 hours

	tWin := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		tWin = append(tWin, ct.Add(-time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		y = append(y, bias+amp*math.Sin(2.0*math.Pi/86400.0*float64(tWin[i].Unix()+phase)))
	}
	opt := &Options{
		DailyOrders: 3,
	}
	td, err := NewUnivariateDataset(tWin, y)
	if err != nil {
		t.Fatal(err)
	}
	f, err := NewForecast(td, opt)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Fit(); err != nil {
		t.Fatal(err)
	}

	coef, err := f.Coefficients()
	if err != nil {
		t.Fatal(err)
	}
	labels := f.FeatureLabels()
	res := make([]float64, 0, len(labels)+1)
	res = append(res, coef["bias"])
	for _, label := range labels {
		res = append(res, coef[label])
	}

	expected := []float64{
		bias,
		3.04, 3.04,
		0.00, 0.00,
		0.00, 0.00,
	}
	assert.InDeltaSlice(t, expected, res, 0.1)
}
