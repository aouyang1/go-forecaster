package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testFitSignal(t *testing.T) (*Forecast, []time.Time, []float64) {
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

	opt := &options.Options{
		SeasonalityOptions: options.SeasonalityOptions{
			SeasonalityConfigs: []options.SeasonalityConfig{
				options.NewDailySeasonalityConfig(3),
			},
		},
	}
	f, err := New(opt)
	require.NoError(t, err)

	err = f.Fit(tWin, y)
	require.NoError(t, err)

	return f, tWin, y
}

func TestFit(t *testing.T) {
	f, _, _ := testFitSignal(t)

	labels, err := f.FeatureLabels()
	require.NoError(t, err)
	res := make([]float64, 0, len(labels))
	intercept, err := f.Intercept()
	require.NoError(t, err)
	res = append(res, intercept)

	coef, err := f.Coefficients()
	require.NoError(t, err)

	for _, label := range labels {
		if label.String() == feature.NewGrowth(feature.GrowthIntercept).String() {
			continue
		}
		res = append(res, coef[label.String()])
	}
	expected := []float64{
		7.90,
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
	f, tWin, y := testFitSignal(t)

	model, err := f.Model()
	require.NoError(t, err)

	// generate new forecast from thhe previous model and perform inference
	fNew, err := NewFromModel(model)
	require.NoError(t, err)

	predicted, _, err := fNew.Predict(tWin)
	require.NoError(t, err)

	labels, err := fNew.FeatureLabels()
	require.NoError(t, err)
	res := make([]float64, 0, len(labels))
	intercept, err := f.Intercept()
	require.NoError(t, err)
	res = append(res, intercept)

	coef, err := fNew.Coefficients()
	require.NoError(t, err)

	for _, label := range labels {
		if label.String() == feature.NewGrowth(feature.GrowthIntercept).String() {
			continue
		}
		res = append(res, coef[label.String()])
	}

	expected := []float64{
		7.90,
		3.72, 2.14,
		0.00, 0.00,
		0.00, 0.00,
	}
	assert.InDeltaSlice(t, expected, res, 0.1)

	scores, err := NewScores(predicted, y)
	require.NoError(t, err)
	assert.Less(t, scores.MSE, 0.0001)
	assert.Less(t, scores.MAPE, 0.0001)
}
