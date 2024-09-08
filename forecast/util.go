package forecast

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecast/changepoint"
	"github.com/aouyang1/go-forecast/feature"
	"gonum.org/v1/gonum/mat"
)

var ErrUnknownTimeFeature = errors.New("unknown time feature")

func ObservationMatrix(y []float64) *mat.Dense {
	n := len(y)
	return mat.NewDense(1, n, y)
}

func generateTimeFeatures(t []time.Time, opt *Options) FeatureSet {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := make(FeatureSet)
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		feat := feature.NewTime("hod")
		tFeat[feat.String()] = Feature{feat, hod}
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		feat := feature.NewTime("dow")
		tFeat[feat.String()] = Feature{feat, dow}
	}
	return tFeat
}

func generateFourierFeatures(tFeat FeatureSet, opt *Options) (FeatureSet, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := make(FeatureSet)
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

func generateFourierOrders(tFeatures FeatureSet, col string, orders []int, period float64) (FeatureSet, error) {
	tFeat, exists := tFeatures[feature.NewTime(col).String()]
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := make(FeatureSet)
	for _, order := range orders {
		sinFeat, cosFeat := generateFourierComponent(tFeat.Data, order, period)
		sinFeatCol := feature.NewSeasonality(col, feature.FourierCompSin, order)
		cosFeatCol := feature.NewSeasonality(col, feature.FourierCompCos, order)
		x[sinFeatCol.String()] = Feature{sinFeatCol, sinFeat}
		x[cosFeatCol.String()] = Feature{cosFeatCol, cosFeat}
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

func generateChangepointFeatures(t []time.Time, chpts []changepoint.Changepoint) FeatureSet {
	var minTime, maxTime time.Time
	for _, tPnt := range t {
		if minTime.IsZero() || tPnt.Before(minTime) {
			minTime = tPnt
		}
		if maxTime.IsZero() || tPnt.After(maxTime) {
			maxTime = tPnt
		}
	}

	sort.Slice(
		chpts,
		func(i, j int) bool {
			return chpts[i].T.Before(chpts[j].T)
		},
	)
	chptStart := len(chpts)
	chptEnd := -1
	for i := 0; i < len(chpts); i++ {
		// haven't reached a changepoint in the time window
		if chpts[i].T.Before(minTime) {
			continue
		}
		if i < chptStart {
			chptStart = i
		}

		// reached end of time window so break
		if chpts[i].T.Equal(maxTime) || chpts[i].T.After(maxTime) {
			chptEnd = i
			break
		}
	}
	if chptEnd == -1 {
		chptEnd = len(chpts)
	}
	fChpts := chpts[chptStart:chptEnd]
	chptFeatures := make([][]float64, len(fChpts)*2)
	for i := 0; i < len(fChpts)*2; i++ {
		chpt := make([]float64, len(t))
		chptFeatures[i] = chpt
	}

	bias := 1.0
	var slope float64
	for i := 0; i < len(t); i++ {
		for j := 0; j < len(fChpts); j++ {
			var beforeNextChpt bool
			var deltaT float64
			if j != len(fChpts)-1 {
				beforeNextChpt = t[i].Before(fChpts[j+1].T)
				deltaT = fChpts[j+1].T.Sub(fChpts[j].T).Seconds()
			} else {
				beforeNextChpt = true
				deltaT = maxTime.Sub(fChpts[j].T).Seconds()
			}
			if t[i].Equal(fChpts[j].T) || (t[i].After(fChpts[j].T) && beforeNextChpt) {
				slope = t[i].Sub(fChpts[j].T).Seconds() / deltaT
				chptFeatures[j*2][i] = bias
				chptFeatures[j*2+1][i] = slope
			}
		}
	}

	feat := make(FeatureSet)
	for i := 0; i < len(fChpts); i++ {
		chpntName := strconv.Itoa(i)
		if fChpts[i].Name != "" {
			chpntName = fChpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		chpntSlope := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)

		feat[chpntBias.String()] = Feature{chpntBias, chptFeatures[i*2]}
		feat[chpntSlope.String()] = Feature{chpntSlope, chptFeatures[i*2+1]}
	}
	return feat
}
