package models

import (
	"testing"

	mat_ "github.com/aouyang1/go-forecaster/mat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestOLSOptionsValidate(t *testing.T) {
	testData := map[string]struct {
		opt      *OLSOptions
		err      error
		expected *OLSOptions
	}{
		"nil": {nil, nil, NewDefaultOLSOptions()},
		"valid": {
			&OLSOptions{
				FitIntercept: true,
			}, nil,
			&OLSOptions{
				FitIntercept: true,
			},
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

func TestOLSRegression(t *testing.T) {
	tol := 1e-5
	testData := map[string]struct {
		x         [][]float64
		y         []float64
		opt       *OLSOptions
		intercept float64
		coef      []float64
	}{
		"ols model intercept": {
			x: [][]float64{
				{0, 0},
				{3, 5},
				{9, 20},
				{12, 6},
				{15, 10},
			},
			y:         []float64{2, 31, 109, 62, 87},
			intercept: 2.0,
			coef:      []float64{3.0, 4.0},
		},
		"ols model no intercept": {
			x: [][]float64{
				{1, 0, 0},
				{1, 3, 5},
				{1, 9, 20},
				{1, 12, 6},
				{1, 15, 10},
			},
			y: []float64{2, 31, 109, 62, 87},
			opt: &OLSOptions{
				FitIntercept: false,
			},
			intercept: 0.0,
			coef:      []float64{2.0, 3.0, 4.0},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			x, err := mat_.NewDenseFromArray(td.x)
			require.Nil(t, err)

			y := mat.NewDense(len(td.y), 1, td.y)

			model, err := NewOLSRegression(td.opt)
			require.Nil(t, err)

			testModel(t, model, x, y, td.intercept, td.coef, tol)
		})
	}
}

func BenchmarkOLSRegression(b *testing.B) {
	x, y, err := generateBenchData(1000, 100)
	if err != nil {
		b.Fatal(err)
	}

	for i := 0; i < b.N; i++ {
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
