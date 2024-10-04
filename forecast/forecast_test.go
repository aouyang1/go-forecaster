package forecast

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
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
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Fit(tWin, y); err != nil {
		t.Fatal(err)
	}

	labels := f.FeatureLabels()
	res := make([]float64, 0, len(labels)+1)
	res = append(res, f.Intercept())

	coef, err := f.Coefficients()
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Fit(tWin, y); err != nil {
		t.Fatal(err)
	}

	model := f.Model()

	// generate new forecast from thhe previous model and perform inference
	f, err = NewFromModel(model)
	if err != nil {
		t.Fatal(err)
	}

	predicted, _, err := f.Predict(tWin)
	if err != nil {
		t.Fatal(err)
	}

	labels := f.FeatureLabels()
	res := make([]float64, 0, len(labels)+1)
	res = append(res, f.Intercept())

	coef, err := f.Coefficients()
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	assert.Less(t, scores.MSE, 0.0001)
	assert.Less(t, scores.MAPE, 0.0001)
}
