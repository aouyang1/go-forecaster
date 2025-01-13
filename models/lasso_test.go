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
	tol := 1e-5
	desTol := 1e-6
	lambda := 0.0
	testData := map[string]struct {
		x         [][]float64
		y         []float64
		opt       *LassoOptions
		intercept float64
		coef      []float64
	}{
		"model intercept": {
			x: [][]float64{
				{0, 0},
				{3, 5},
				{9, 20},
				{12, 6},
				{15, 10},
			},
			y: []float64{2, 31, 109, 62, 87},
			opt: func() *LassoOptions {
				opt := NewDefaultLassoOptions()
				opt.Lambda = lambda
				opt.Tolerance = desTol
				return opt
			}(),
			intercept: 2.0,
			coef:      []float64{3.0, 4.0},
		},
		"model no intercept": {
			x: [][]float64{
				{1, 0, 0},
				{1, 3, 5},
				{1, 9, 20},
				{1, 12, 6},
				{1, 15, 10},
			},
			y: []float64{2, 31, 109, 62, 87},
			opt: func() *LassoOptions {
				opt := NewDefaultLassoOptions()
				opt.Lambda = lambda
				opt.Tolerance = desTol
				opt.FitIntercept = false
				return opt
			}(),
			intercept: 0.0,
			coef:      []float64{2.0, 3.0, 4.0},
		},
		"model constant": {
			x: [][]float64{
				{1},
				{1},
				{1},
				{1},
				{1},
			},
			y: []float64{3, 3, 3, 3, 3},
			opt: func() *LassoOptions {
				opt := NewDefaultLassoOptions()
				opt.Lambda = lambda
				opt.Tolerance = desTol
				opt.FitIntercept = false
				return opt
			}(),
			intercept: 0.0,
			coef:      []float64{3.0},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			x, err := mat_.NewDenseFromArray(td.x)
			require.Nil(t, err)

			y := mat.NewDense(len(td.y), 1, td.y)

			model, err := NewLassoRegression(td.opt)
			require.Nil(t, err)

			testModel(t, model, x, y, td.intercept, td.coef, tol)
		})
	}
}

func TestLassoAutoRegression(t *testing.T) {
	// y = 2 + 3*x0 + 4*x1
	tol := 1e-5
	desTol := 1e-6
	lambdas := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	parallelization := 4

	testData := map[string]struct {
		x         [][]float64
		y         []float64
		opt       *LassoAutoOptions
		intercept float64
		coef      []float64
	}{
		"auto model intercept": {
			x: [][]float64{
				{0, 0},
				{3, 5},
				{9, 20},
				{12, 6},
				{15, 10},
			},
			y: []float64{2, 31, 109, 62, 87},
			opt: func() *LassoAutoOptions {
				opt := NewDefaultLassoAutoOptions()
				opt.Lambdas = lambdas
				opt.Tolerance = desTol
				opt.FitIntercept = true
				opt.Parallelization = parallelization
				return opt
			}(),
			intercept: 2.0,
			coef:      []float64{3.0, 4.0},
		},
		"auto model no intercept": {
			x: [][]float64{
				{1, 0, 0},
				{1, 3, 5},
				{1, 9, 20},
				{1, 12, 6},
				{1, 15, 10},
			},
			y: []float64{2, 31, 109, 62, 87},
			opt: func() *LassoAutoOptions {
				opt := NewDefaultLassoAutoOptions()
				opt.Lambdas = lambdas
				opt.Tolerance = desTol
				opt.FitIntercept = false
				opt.Parallelization = parallelization
				return opt
			}(),
			intercept: 0.0,
			coef:      []float64{2.0, 3.0, 4.0},
		},
		"auto model constant": {
			x: [][]float64{
				{1},
				{1},
				{1},
				{1},
				{1},
			},
			y: []float64{3, 3, 3, 3, 3},
			opt: func() *LassoAutoOptions {
				opt := NewDefaultLassoAutoOptions()
				opt.Lambdas = lambdas
				opt.Tolerance = desTol
				opt.FitIntercept = false
				opt.Parallelization = parallelization
				return opt
			}(),
			intercept: 0.0,
			coef:      []float64{3.0},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			x, err := mat_.NewDenseFromArray(td.x)
			require.Nil(t, err)

			y := mat.NewDense(len(td.y), 1, td.y)

			model, err := NewLassoAutoRegression(td.opt)
			require.Nil(t, err)

			testModel(t, model, x, y, td.intercept, td.coef, tol)
		})
	}
}

func BenchmarkLassoRegression(b *testing.B) {
	x, y, err := generateBenchData(1000, 100)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
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
