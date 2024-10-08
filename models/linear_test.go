package models

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestOLS(t *testing.T) {
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

	intercept, coef := OLS(mObs, mY)
	assert.InDelta(t, 2.0, intercept, 0.00001)
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)
}

func TestLassoRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
	obs := [][]float64{
		{1, 1, 1, 1},
		{0, 3, 9, 12},
		{0, 5, 20, 6},
	}
	y := []float64{2, 31, 109, 62}

	opt := NewDefaultLassoOptions()
	opt.Lambda = 0
	opt.Tolerance = 1e-6

	intercept, coef, err := LassoRegression(obs, y, opt)
	require.Nil(t, err)
	assert.InDelta(t, 2.0, intercept, 0.00001)
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)
}

func BenchmarkOLS(b *testing.B) {
	nObs := 1000
	nFeat := 100

	data := make([]float64, 0, nObs*nFeat)
	for i := 0; i < cap(data); i++ {
		val := float64(i)
		if i%nFeat == 0 {
			val = 1.0
		}
		data = append(data, val)
	}

	data2 := make([]float64, 0, nObs)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		mObs := mat.NewDense(nObs, nFeat, data)
		mY := mat.NewDense(1, nObs, data2)

		OLS(mObs, mY)
	}
}

func BenchmarkLassoRegression(b *testing.B) {
	nObs := 1000
	nFeat := 100
	data := make([][]float64, 0, nFeat)
	for i := 0; i < nFeat; i++ {
		feat := make([]float64, nObs)
		for j := 0; j < nObs; j++ {
			val := float64(j*nObs + i)
			if i == 0 {
				val = 1.0
			}
			feat[j] = val
		}
		data = append(data, feat)
	}

	data2 := make([]float64, 0, nObs)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		mObs := data
		mY := data2

		LassoRegression(mObs, mY, nil)
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
