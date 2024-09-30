package forecast

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/feature"
	"gonum.org/v1/gonum/mat"
)

var ErrUnknownTimeFeature = errors.New("unknown time feature")

func ObservationMatrix(y []float64) *mat.Dense {
	n := len(y)
	return mat.NewDense(1, n, y)
}

func generateTimeFeatures(t []time.Time, opt *Options) feature.Set {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := make(feature.Set)
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		feat := feature.NewTime("hod")
		tFeat[feat.String()] = feature.Data{
			F:    feat,
			Data: hod,
		}
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		feat := feature.NewTime("dow")
		tFeat[feat.String()] = feature.Data{
			F:    feat,
			Data: dow,
		}
	}
	return tFeat
}

func generateFourierFeatures(tFeat feature.Set, opt *Options) (feature.Set, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := make(feature.Set)
	if opt.DailyOrders > 0 {
		var orders []int
		for i := 1; i <= opt.DailyOrders; i++ {
			orders = append(orders, i)
		}
		dailyFeatures, err := generateFourierOrders(tFeat, "hod", orders, 24.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "hod", err)
		}
		for label, features := range dailyFeatures {
			x[label] = features
		}
	}

	if opt.WeeklyOrders > 0 {
		var orders []int
		for i := 1; i <= opt.WeeklyOrders; i++ {
			if i%7 == 0 && i/7 <= opt.DailyOrders {
				// colinear feature so skip
				continue
			}
			orders = append(orders, i)
		}
		weeklyFeatures, err := generateFourierOrders(tFeat, "dow", orders, 7.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "dow", err)
		}
		for label, features := range weeklyFeatures {
			x[label] = features
		}
	}
	return x, nil
}

func generateFourierOrders(tFeatures feature.Set, col string, orders []int, period float64) (feature.Set, error) {
	tFeat, exists := tFeatures[feature.NewTime(col).String()]
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := make(feature.Set)
	for _, order := range orders {
		sinFeat, cosFeat := generateFourierComponent(tFeat.Data, order, period)
		sinFeatCol := feature.NewSeasonality(col, feature.FourierCompSin, order)
		cosFeatCol := feature.NewSeasonality(col, feature.FourierCompCos, order)
		x[sinFeatCol.String()] = feature.Data{
			F:    sinFeatCol,
			Data: sinFeat,
		}
		x[cosFeatCol.String()] = feature.Data{
			F:    cosFeatCol,
			Data: cosFeat,
		}
	}

	return x, nil
}

func generateFourierComponent(timeFeature []float64, order int, period float64) ([]float64, []float64) {
	omega := 2.0 * math.Pi * float64(order) / period
	sinFeat := make([]float64, len(timeFeature))
	cosFeat := make([]float64, len(timeFeature))
	for i, tFeat := range timeFeature {
		rad := omega * tFeat
		sinFeat[i] = math.Sin(rad)
		cosFeat[i] = math.Cos(rad)
	}
	return sinFeat, cosFeat
}

func generateAutoChangepoints(t []time.Time, n int) []changepoint.Changepoint {
	var minTime, maxTime time.Time
	for _, tPnt := range t {
		if minTime.IsZero() || tPnt.Before(minTime) {
			minTime = tPnt
		}
		if maxTime.IsZero() || tPnt.After(maxTime) {
			maxTime = tPnt
		}
	}

	window := maxTime.Sub(minTime)
	changepointWinNs := int64(window.Nanoseconds()) / int64(n)
	chpts := make([]changepoint.Changepoint, 0, n)

	for i := 0; i < n; i++ {
		chpntTime := minTime.Add(time.Duration(changepointWinNs * int64(i)))
		chpts = append(
			chpts,
			changepoint.New("auto_"+strconv.Itoa(i), chpntTime),
		)
	}
	return chpts
}

func generateChangepointFeatures(t []time.Time, chpts []changepoint.Changepoint) feature.Set {
	var minTime, maxTime time.Time
	for _, tPnt := range t {
		if minTime.IsZero() || tPnt.Before(minTime) {
			minTime = tPnt
		}
		if maxTime.IsZero() || tPnt.After(maxTime) {
			maxTime = tPnt
		}
	}

	chptFeatures := make([][]float64, len(chpts)*2)
	for i := 0; i < len(chpts)*2; i++ {
		chpt := make([]float64, len(t))
		chptFeatures[i] = chpt
	}

	bias := 1.0
	var slope float64
	for i := 0; i < len(t); i++ {
		for j := 0; j < len(chpts); j++ {
			if t[i].Equal(chpts[j].T) || t[i].After(chpts[j].T) {
				deltaT := maxTime.Sub(chpts[j].T).Seconds()
				slope = t[i].Sub(chpts[j].T).Seconds() / deltaT
				chptFeatures[j*2][i] = bias
				chptFeatures[j*2+1][i] = slope
			}
		}
	}

	return makeChangepointFeatureSet(chpts, chptFeatures)
}

func makeChangepointFeatureSet(chpts []changepoint.Changepoint, chptFeatures [][]float64) feature.Set {
	feat := make(feature.Set)
	for i := 0; i < len(chpts); i++ {
		chpntName := strconv.Itoa(i)
		if chpts[i].Name != "" {
			chpntName = chpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		chpntSlope := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)

		feat[chpntBias.String()] = feature.Data{
			F:    chpntBias,
			Data: chptFeatures[i*2],
		}
		feat[chpntSlope.String()] = feature.Data{
			F:    chpntSlope,
			Data: chptFeatures[i*2+1],
		}
	}
	return feat
}
