package forecast

import (
	"testing"

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

	intercept, coef := OLS(mObs, mY)
	assert.InDelta(t, 2.0, intercept, 0.00001)
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)
}

func TestCoordinateDescent(t *testing.T) {
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

	intercept, coef := CoordinateDescent(mObs, mY, 0, 100)
	assert.InDelta(t, 2.0, intercept, 0.00001)
	assert.InDelta(t, 3.0, coef[0], 0.00001)
	assert.InDelta(t, 4.0, coef[1], 0.00001)
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

		OLS(mObs, mY)
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
