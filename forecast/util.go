package forecast

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/feature"
)

var ErrUnknownTimeFeature = errors.New("unknown time feature")

func generateTimeFeatures(t []time.Time, opt *Options) *feature.Set {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := feature.NewSet()
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		feat := feature.NewTime("hod")
		tFeat.Set(feat, hod)
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		feat := feature.NewTime("dow")
		tFeat.Set(feat, dow)
	}
	return tFeat
}

func generateFourierFeatures(tFeat *feature.Set, opt *Options) (*feature.Set, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := feature.NewSet()
	if opt.DailyOrders > 0 {
		var orders []int
		for i := 1; i <= opt.DailyOrders; i++ {
			orders = append(orders, i)
		}
		dailyFeatures, err := generateFourierOrders(tFeat, "hod", orders, 24.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "hod", err)
		}
		x.Update(dailyFeatures)
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
		x.Update(weeklyFeatures)
	}
	return x, nil
}

func generateFourierOrders(tFeatures *feature.Set, col string, orders []int, period float64) (*feature.Set, error) {
	if tFeatures == nil {
		return nil, ErrUnknownTimeFeature
	}
	tFeat, exists := tFeatures.Get(feature.NewTime(col))
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := feature.NewSet()
	for _, order := range orders {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := feature.NewSeasonality(col, feature.FourierCompSin, order)
		cosFeatCol := feature.NewSeasonality(col, feature.FourierCompCos, order)
		x.Set(sinFeatCol, sinFeat)
		x.Set(cosFeatCol, cosFeat)
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

func generateChangepointFeatures(t []time.Time, chpts []changepoint.Changepoint, trainingEndTime time.Time) *feature.Set {
	filteredChpts := make([]changepoint.Changepoint, 0, len(chpts))
	for _, chpt := range chpts {
		// skip over changepoints that are after the training end time since they'll
		// not have been modeled producing zeroes in the feature set
		if chpt.T.After(trainingEndTime) {
			continue
		}
		filteredChpts = append(filteredChpts, chpt)
	}

	// create a slice of features where it goes in the order of bias, slope for each changepoint
	chptFeatures := make([][]float64, len(filteredChpts)*2)
	for i := 0; i < len(filteredChpts)*2; i++ {
		chpt := make([]float64, len(t))
		chptFeatures[i] = chpt
	}

	// compute dt between training end time and changepoint time
	deltaT := make([]float64, len(filteredChpts))
	for i, chpt := range filteredChpts {
		deltaT[i] = trainingEndTime.Sub(chpt.T).Seconds()
	}

	bias := 1.0
	var slope float64
	for i := 0; i < len(t); i++ {
		for j := 0; j < len(filteredChpts); j++ {
			if t[i].Equal(filteredChpts[j].T) || t[i].After(filteredChpts[j].T) {
				slope = t[i].Sub(filteredChpts[j].T).Seconds() / deltaT[j]
				chptFeatures[j*2][i] = bias
				chptFeatures[j*2+1][i] = slope
			}
		}
	}

	return makeChangepointFeatureSet(filteredChpts, chptFeatures)
}

func makeChangepointFeatureSet(chpts []changepoint.Changepoint, chptFeatures [][]float64) *feature.Set {
	feat := feature.NewSet()
	for i := 0; i < len(chpts); i++ {
		chpntName := strconv.Itoa(i)
		if chpts[i].Name != "" {
			chpntName = chpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		chpntSlope := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)

		feat.Set(chpntBias, chptFeatures[i*2])
		feat.Set(chpntSlope, chptFeatures[i*2+1])
	}
	return feat
}
