package main

import (
	"fmt"
	"math"
	"time"
)

func main() {
	// create a daily sine wave at minutely with one week
	minutes := 24 * 60
	// minutes = 60
	t := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		t = append(t, ct.Add(-time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		y = append(y, 1.2+4.3*math.Sin(2.0*math.Pi/86400.0*float64(t[i].Unix()+3*60*60)))
	}
	opt := &Options{
		DailyOrders: 1,
	}
	td, err := NewUnivariateDataset(t, y)
	if err != nil {
		panic(err)
	}
	f, err := NewForecast(opt)
	if err != nil {
		panic(err)
	}
	if err := f.Fit(td); err != nil {
		panic(err)
	}
	eq, err := f.ModelEq()
	if err != nil {
		panic(err)
	}
	fmt.Println(eq)
}
