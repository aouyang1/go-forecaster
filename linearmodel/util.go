package linearmodel

import (
	"math"
	"testing"
	"time"

	mat_ "github.com/aouyang1/go-forecaster/mat"

	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func testModel(t *testing.T, model Model, x, y mat.Matrix, intercept float64, coef []float64, tol float64) {
	err := model.Fit(x, y)
	require.Nil(t, err)

	assert.InDelta(t, intercept, model.Intercept(), tol, "intercept")

	c := model.Coef()
	assert.InDeltaSlice(t, coef, c, tol, "coefficients")

	r2, err := model.Score(x, y)
	require.Nil(t, err)
	assert.InDelta(t, 1.0, r2, tol, "score")
}

func generateBenchData(minutes, nFeat int) (mat.Matrix, mat.Matrix, error) {
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	out := make(timedataset.Series, minutes)

	period := 86400.0
	out.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateWaveY(t, 23.4, period, 7.0, 6.0*60*60+period/2.0/2.0/3.0).MaskWithTimeRange(t[minutes*4/16], t[minutes*5/16], t)).
		Add(timedataset.GenerateWaveY(t, -7.3, period, 3.0, 2*60*60+period/2.0/2.0/3.0).MaskWithWeekend(t)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0))

	epoch := make([]float64, len(t))
	for i, tPnt := range t {
		epochNano := float64(tPnt.UnixNano()) / 1e9
		epoch[i] = epochNano
	}

	data := make([][]float64, minutes)
	for i := range minutes {
		obs := make([]float64, nFeat*2+1)
		obs[0] = 1.0
		data[i] = obs
	}
	for order := 1; order <= nFeat; order++ {
		omega := 2.0 * math.Pi * float64(order) / period
		for i, tFeat := range epoch {
			rad := omega * tFeat
			data[i][2*order-1] = math.Sin(rad)
			data[i][2*order] = math.Cos(rad)
		}
	}

	x, err := mat_.NewDenseFromArray(data)
	if err != nil {
		return nil, nil, err
	}

	y := mat.NewDense(len(out), 1, out)
	return x, y, nil
}
