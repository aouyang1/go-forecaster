package models

import (
	"testing"

	mat_ "github.com/aouyang1/go-forecaster/mat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func testModel(t *testing.T, model Model, x, y mat.Matrix, intercept float64, coef []float64, tol float64) {
	err := model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, intercept, model.Intercept(), tol)

	c := model.Coef()
	assert.InDeltaSlice(t, coef, c, tol)

	r2, err := model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, tol)
}

func generateBenchData(nObs, nFeat int) (mat.Matrix, mat.Matrix, error) {
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

	x, err := mat_.NewDenseFromArray(data)
	if err != nil {
		return nil, nil, err
	}

	y := mat.NewDense(nObs, 1, data2)
	return x, y, nil
}
