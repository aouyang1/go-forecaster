package main

import (
	"testing"

	"github.com/sajari/regression"
	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestLinearRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
	obs := []float64{
		1, 0, 0,
		1, 3, 5,
		1, 9, 20,
		1, 12, 6,
	}
	y := []float64{2, 31, 109, 62}

	mObs := mat.NewDense(4, 3, obs)
	mY := mat.NewDense(1, 4, y)

	coef := Regression(mObs, mY)
	assert.InDelta(t, 2.0, coef[0], 0.00001)
	assert.InDelta(t, 3.0, coef[1], 0.00001)
	assert.InDelta(t, 4.0, coef[2], 0.00001)
}

func TestLibRegression(t *testing.T) {
	r := new(regression.Regression)
	r.SetObserved("y")
	r.SetVar(0, "x0")
	r.SetVar(1, "x1")
	r.Train(
		regression.DataPoint(2, []float64{0, 0}),
		regression.DataPoint(31, []float64{3, 5}),
		regression.DataPoint(109, []float64{9, 20}),
		regression.DataPoint(62, []float64{12, 6}),
	)
	r.Run()

	t.Logf("Regression formula:\n%v\n", r.Formula)
	t.Logf("Regression:\n%s\n", r)
	assert.InDeltaSlice(t, []float64{2.0, 3.0, 4.0}, r.GetCoeffs(), 0.00001)
}

func BenchmarkRegression(b *testing.B) {
	data := make([]float64, 0, 300)
	for i := 0; i < cap(data); i++ {
		val := float64(i)
		if i%5 == 0 {
			val = 1.0
		}
		data = append(data, val)
	}

	data2 := make([]float64, 0, 60)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		mObs := mat.NewDense(60, 5, data)
		mY := mat.NewDense(1, 60, data2)

		Regression(mObs, mY)
	}
}

func BenchmarkRegression2(b *testing.B) {
	data := make(regression.DataPoints, 0, 60)
	for i := 0; i < cap(data); i++ {
		ifloat := float64(i)
		obs := []float64{ifloat*5 + 1, ifloat*5 + 2, ifloat*5 + 3, ifloat*5 + 4}
		point := regression.DataPoint(ifloat, obs)
		data = append(data, point)
	}

	for i := 0; i < b.N; i++ {
		r := new(regression.Regression)
		r.SetObserved("y")
		r.SetVar(0, "x0")
		r.SetVar(1, "x1")
		r.SetVar(2, "x2")
		r.SetVar(3, "x3")
		r.Train(data...)

		if err := r.Run(); err != nil {
			b.Error(err)
		}
		r.GetCoeffs()
	}
}

func BenchmarkMatMultObservationsByWeight(b *testing.B) {
	data := make([]float64, 0, 30000)
	for i := 0; i < cap(data); i++ {
		data = append(data, float64(i))
	}

	data2 := make([]float64, 0, 1000)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		ma := mat.NewDense(30, 1000, data)
		mb := mat.NewDense(1000, 1, data2)
		var mc mat.Dense
		mc.Mul(ma, mb)
	}
}

func BenchmarkMatMultWeightByObservations(b *testing.B) {
	data := make([]float64, 0, 1000)
	for i := 0; i < cap(data); i++ {
		data = append(data, float64(i))
	}

	data2 := make([]float64, 0, 30000)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		ma := mat.NewDense(1, 1000, data)
		mb := mat.NewDense(1000, 30, data2)
		var mc mat.Dense
		mc.Mul(ma, mb)
	}
}
