package models

import (
	"testing"

	mat_ "github.com/aouyang1/go-forecaster/mat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestLassoOptionsValidate(t *testing.T) {
	testData := map[string]struct {
		opt      *LassoOptions
		err      error
		expected *LassoOptions
	}{
		"nil": {nil, nil, NewDefaultLassoOptions()},
		"valid": {
			&LassoOptions{
				Lambda:     1.0,
				Iterations: 100,
				Tolerance:  1e-5,
			}, nil,
			&LassoOptions{
				Lambda:     1.0,
				Iterations: 100,
				Tolerance:  1e-5,
			},
		},
		"invalid lambda": {
			&LassoOptions{Lambda: -1.0},
			ErrNegativeLambda, nil,
		},
		"invalid iterations": {
			&LassoOptions{Iterations: -1.0},
			ErrNegativeIterations, nil,
		},
		"invalid tolerance": {
			&LassoOptions{Tolerance: -1.0},
			ErrNegativeTolerance, nil,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			opt, err := td.opt.Validate()
			if td.err != nil {
				assert.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)
			assert.Equal(t, td.expected, opt)
		})
	}
}

func TestLassoRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
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

	opt = NewDefaultLassoOptions()
	opt.Lambda = 0
	opt.Tolerance = 1e-6
	opt.FitIntercept = false

	model, err = NewLassoRegression(opt)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	coef = model.Coef()
	assert.InDelta(t, 2.0, coef[0], 0.00001)
	assert.InDelta(t, 3.0, coef[1], 0.00001)
	assert.InDelta(t, 4.0, coef[2], 0.00001)

	r2, err = model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)

	x, err = mat_.NewDenseFromArray(
		[][]float64{
			{1},
			{1},
			{1},
			{1},
		},
	)
	require.Nil(t, err)

	y = mat.NewDense(4, 1, []float64{3, 3, 3, 3})

	opt = NewDefaultLassoOptions()
	opt.Lambda = 0
	opt.Tolerance = 1e-6
	opt.FitIntercept = false

	model, err = NewLassoRegression(opt)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	coef = model.Coef()
	assert.InDelta(t, 3.0, coef[0], 0.00001)

	r2, err = model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)
}

func TestLassoAutoRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
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

	opt := NewDefaultLassoAutoOptions()
	opt.Lambdas = []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt.Tolerance = 1e-6
	opt.FitIntercept = true
	opt.Parallelization = 4

	model, err := NewLassoAutoRegression(opt)
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

	opt = NewDefaultLassoAutoOptions()
	opt.Lambdas = []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt.Tolerance = 1e-6
	opt.FitIntercept = false
	opt.Parallelization = 4

	model, err = NewLassoAutoRegression(opt)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	coef = model.Coef()
	assert.InDelta(t, 2.0, coef[0], 0.00001)
	assert.InDelta(t, 3.0, coef[1], 0.00001)
	assert.InDelta(t, 4.0, coef[2], 0.00001)

	r2, err = model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)

	x, err = mat_.NewDenseFromArray(
		[][]float64{
			{1},
			{1},
			{1},
			{1},
		},
	)
	require.Nil(t, err)

	y = mat.NewDense(4, 1, []float64{3, 3, 3, 3})

	opt = NewDefaultLassoAutoOptions()
	opt.Lambdas = []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt.Tolerance = 1e-6
	opt.FitIntercept = false
	opt.Parallelization = 4

	model, err = NewLassoAutoRegression(opt)
	require.Nil(t, err)

	err = model.Fit(x, y)
	require.Nil(t, err)

	coef = model.Coef()
	assert.InDelta(t, 3.0, coef[0], 0.00001)

	r2, err = model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, 0.00001)
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
		x, err := mat_.NewDenseFromArray(data)
		if err != nil {
			b.Error(err)
			continue
		}
		y := mat.NewDense(nObs, 1, data2)
		opt := NewDefaultLassoOptions()
		opt.FitIntercept = false
		model, err := NewLassoRegression(opt)
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
