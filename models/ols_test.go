package models

import (
	"testing"

	mat_ "github.com/aouyang1/go-forecaster/mat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestOLSRegression(t *testing.T) {
	x, err := mat_.NewDenseFromArray(
		[][]float64{
			{0, 0},
			{3, 5},
			{9, 20},
			{12, 6},
		},
	)
	require.Nil(t, err)

	y := mat.NewDense(4, 1, []float64{2, 31, 109, 62})

	model, err := NewOLSRegression(nil)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, 2.0, model.Intercept(), 0.00001)

	coef := model.Coef()
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)

	r2, err := model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)

	x, err = mat_.NewDenseFromArray(
		[][]float64{
			{1, 0, 0},
			{1, 3, 5},
			{1, 9, 20},
			{1, 12, 6},
		},
	)
	require.Nil(t, err)

	y = mat.NewDense(4, 1, []float64{2, 31, 109, 62})

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

	r2, err = model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)
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
		x, err := mat_.NewDenseFromArray(data)
		if err != nil {
			b.Error(err)
			continue
		}
		y := mat.NewDense(nObs, 1, data2)
		model, err := NewOLSRegression(
			&OLSOptions{
				FitIntercept: false,
			},
		)
		if err != nil {
			b.Error(err)
			continue
		}
		if err := model.Fit(x, y); err != nil {
			b.Error(err)
			continue
		}
	}
}
