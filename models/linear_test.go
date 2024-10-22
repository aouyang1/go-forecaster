package models

import (
	"testing"

	"github.com/aouyang1/go-forecaster/array"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestOLSRegression(t *testing.T) {
	x, err := array.New2D(
		[][]float64{
			{0, 0},
			{3, 5},
			{9, 20},
			{12, 6},
		},
	)
	require.Nil(t, err)

	y := array.New1D([]float64{2, 31, 109, 62})

	/*
		mObs := mat.NewDense(4, 3, xArr.Flatten())
		mY := mat.NewDense(1, 4, y.Flatten())

		intercept, coef := OLS(mObs, mY)
	*/
	model, err := NewOLSRegression(nil)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, 2.0, model.Intercept(), 0.00001)

	coef := model.Coef()
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)

	x, err = array.New2D(
		[][]float64{
			{1, 0, 0},
			{1, 3, 5},
			{1, 9, 20},
			{1, 12, 6},
		},
	)
	require.Nil(t, err)

	y = array.New1D([]float64{2, 31, 109, 62})

	model, err = NewOLSRegression(
		&OLSOptions{
			FitIntercept: false,
		},
	)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, 0.0, model.Intercept(), 0.00001)

	coef = model.Coef()
	assert.InDelta(t, 2.0, coef[0], 0.00001)
	assert.InDelta(t, 3.0, coef[1], 0.00001)
	assert.InDelta(t, 4.0, coef[2], 0.00001)
}

func TestLassoRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
	x, err := array.New2D(
		[][]float64{
			{0, 0},
			{3, 5},
			{9, 20},
			{12, 6},
		},
	)
	require.Nil(t, err)

	y := array.New1D([]float64{2, 31, 109, 62})

	opt := NewDefaultLassoOptions()
	opt.Lambda = 0
	opt.Tolerance = 1e-6

	model, err := NewLassoRegression(opt)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, 2.0, model.Intercept(), 0.00001)

	coef := model.Coef()
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)
}

func BenchmarkOLSRegression(b *testing.B) {
	nObs := 1000
	nFeat := 100

	data := make([][]float64, nObs)
	for i := 0; i < nObs; i++ {
		data[i] = make([]float64, nFeat)
		for j := 0; j < nFeat; j++ {
			val := float64(i*nFeat + j)
			if j == 0 {
				val = 1.0
			}
			data[i][j] = val
		}
	}

	data2 := make([]float64, 0, nObs)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		xArr, err := array.New2D(data)
		if err != nil {
			b.Error(err)
			continue
		}
		yArr := array.New1D(data2)
		model, err := NewOLSRegression(
			&OLSOptions{
				FitIntercept: false,
			},
		)
		if err != nil {
			b.Error(err)
			continue
		}
		if err := model.Fit(xArr, yArr); err != nil {
			b.Error(err)
			continue
		}
	}
}

func BenchmarkLassoRegression(b *testing.B) {
	nObs := 1000
	nFeat := 100

	data := make([][]float64, nObs)
	for i := 0; i < nObs; i++ {
		data[i] = make([]float64, nFeat)
		for j := 0; j < nFeat; j++ {
			val := float64(i*nFeat + j)
			if j == 0 {
				val = 1.0
			}
			data[i][j] = val
		}
	}

	data2 := make([]float64, 0, nObs)
	for i := 0; i < cap(data2); i++ {
		data2 = append(data2, float64(i))
	}

	for i := 0; i < b.N; i++ {
		xArr, err := array.New2D(data)
		if err != nil {
			b.Error(err)
			continue
		}
		yArr := array.New1D(data2)
		opt := NewDefaultLassoOptions()
		opt.FitIntercept = false
		model, err := NewLassoRegression(opt)
		if err != nil {
			b.Error(err)
			continue
		}
		if err := model.Fit(xArr, yArr); err != nil {
			b.Error(err)
			continue
		}
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
