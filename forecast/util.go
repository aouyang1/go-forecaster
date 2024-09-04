package forecast

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecast/changepoint"
	"github.com/aouyang1/go-forecast/feature"
	"gonum.org/v1/gonum/mat"
)

func featureMatrix(t []time.Time, featureLabels []feature.Feature, features map[feature.Feature][]float64) *mat.Dense {
	m := len(t)
	n := len(features) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < m; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	for _, label := range featureLabels {
		feature := features[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

func observationMatrix(y []float64) *mat.Dense {
	n := len(y)
	return mat.NewDense(1, n, y)
}

func generateTimeFeatures(t []time.Time, opt *Options) map[feature.Feature][]float64 {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := make(map[feature.Feature][]float64)
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		feat := feature.NewTime("hod")
		tFeat[feat] = hod
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		feat := feature.NewTime("dow")
		tFeat[feat] = dow
	}
	return tFeat
}

func featureLabels(features map[feature.Feature][]float64) []feature.Feature {
	labels := make([]feature.Feature, 0, len(features))
	for label := range features {
		labels = append(labels, label)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i].String() < labels[j].String()
		},
	)
	return labels
}

func generateFourierFeatures(tFeat map[feature.Feature][]float64, opt *Options) (map[feature.Feature][]float64, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := make(map[feature.Feature][]float64)
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

func generateFourierOrders(tFeatures map[feature.Feature][]float64, col string, orders []int, period float64) (map[feature.Feature][]float64, error) {
	tFeat, exists := tFeatures[feature.NewTime(col)]
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := make(map[feature.Feature][]float64)
	for _, order := range orders {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := feature.NewSeasonality(col, feature.FourierCompSin, order)
		cosFeatCol := feature.NewSeasonality(col, feature.FourierCompCos, order)
		x[sinFeatCol] = sinFeat
		x[cosFeatCol] = cosFeat
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

func generateChangepointFeatures(t []time.Time, chpts []changepoint.Changepoint) map[feature.Feature][]float64 {
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

	feat := make(map[feature.Feature][]float64)
	for i := 0; i < len(fChpts); i++ {
		chpntName := strconv.Itoa(i)
		if fChpts[i].Name != "" {
			chpntName = fChpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		chpntSlope := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)

		feat[chpntBias] = chptFeatures[i*2]
		feat[chpntSlope] = chptFeatures[i*2+1]
	}
	return feat
}
