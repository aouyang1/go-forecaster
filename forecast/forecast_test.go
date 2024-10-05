package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFit(t *testing.T) {
	// create a daily sine wave at minutely with one week
	minutes := 7 * 24 * 60
	bias := 7.9
	amp := 4.3
	phase := int64(4 * 60 * 60) // 3 hours

	tWin := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		tWin = append(tWin, ct.Add(time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		y = append(y, bias+amp*math.Sin(2.0*math.Pi/86400.0*float64(tWin[i].Unix()+phase)))
	}
	opt := &Options{
		DailyOrders: 3,
	}
	f, err := New(opt)
	require.Nil(t, err)

	err = f.Fit(tWin, y)
	require.Nil(t, err)

	labels := f.FeatureLabels()
	res := make([]float64, 0, len(labels)+1)
	res = append(res, f.Intercept())

	coef, err := f.Coefficients()
	require.Nil(t, err)

	for _, label := range labels {
		res = append(res, coef[label.String()])
	}

	expected := []float64{
		bias,
		3.72, 2.14,
		0.00, 0.00,
		0.00, 0.00,
	}
	assert.InDeltaSlice(t, expected, res, 0.1)

	scores := f.Scores()
	assert.Less(t, scores.MSE, 0.0001)
	assert.Less(t, scores.MAPE, 0.0001)
}

func TestFitFromModel(t *testing.T) {
	// create a daily sine wave at minutely with one week
	minutes := 7 * 24 * 60
	bias := 7.9
	amp := 4.3
	phase := int64(4 * 60 * 60) // 3 hours

	tWin := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		tWin = append(tWin, ct.Add(time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		y = append(y, bias+amp*math.Sin(2.0*math.Pi/86400.0*float64(tWin[i].Unix()+phase)))
	}
	opt := &Options{
		DailyOrders: 3,
	}
	f, err := New(opt)
	require.Nil(t, err)

	err = f.Fit(tWin, y)
	require.Nil(t, err)

	model, err := f.Model()
	require.Nil(t, err)

	// generate new forecast from thhe previous model and perform inference
	fNew, err := NewFromModel(model)
	require.Nil(t, err)

	predicted, _, err := fNew.Predict(tWin)
	require.Nil(t, err)

	labels := fNew.FeatureLabels()
	res := make([]float64, 0, len(labels)+1)
	res = append(res, f.Intercept())

	coef, err := fNew.Coefficients()
	require.Nil(t, err)

	for _, label := range labels {
		res = append(res, coef[label.String()])
	}

	expected := []float64{
		bias,
		3.72, 2.14,
		0.00, 0.00,
		0.00, 0.00,
	}
	assert.InDeltaSlice(t, expected, res, 0.1)

	scores, err := NewScores(predicted, y)
	require.Nil(t, err)
	assert.Less(t, scores.MSE, 0.0001)
	assert.Less(t, scores.MAPE, 0.0001)
}
