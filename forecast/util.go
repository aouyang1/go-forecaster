package forecast

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecaster/changepoint"
	"github.com/aouyang1/go-forecaster/feature"
	"gonum.org/v1/gonum/floats"
)

var ErrUnknownTimeFeature = errors.New("unknown time feature")

func generateTimeFeatures(t []time.Time, opt *Options) *feature.Set {
	if opt == nil {
		opt = NewDefaultOptions()
	}

	tFeat := feature.NewSet()

	if opt.WeekendOptions.Enabled {
		var locOverride *time.Location
		if opt.WeekendOptions.TimezoneOverride != "" {
			var err error
			locOverride, err = time.LoadLocation(opt.WeekendOptions.TimezoneOverride)
			if err != nil {
				slog.Warn("invalid timezone location override for weekend options using dataset timezone", "timezone_override", opt.WeekendOptions.TimezoneOverride)
			}
		}

		weekend := make([]float64, len(t))
		var wkday time.Weekday
		for i, tPnt := range t {
			if locOverride != nil {
				tPnt = tPnt.In(locOverride)
			}
			wkday = tPnt.Weekday()
			if wkday == time.Saturday || wkday == time.Sunday {
				weekend[i] = 1.0
			}
		}
		feat := feature.NewTime("is_weekend")
		tFeat.Set(feat, weekend)
	}

	for _, e := range opt.EventOptions.Events {
		if err := e.Valid(); err != nil {
			slog.Warn("not separately modelling invalid event", "name", e.Name, "error", err.Error())
			continue
		}

		feat := feature.NewTime(fmt.Sprintf("event_%s", e.Name))
		if _, exists := tFeat.Get(feat); exists {
			slog.Warn("event feature already exists", "event_name", e.Name)
			continue
		}

		eventMask := make([]float64, len(t))
		for i, tPnt := range t {
			if (tPnt.After(e.Start) || tPnt.Equal(e.Start)) && (tPnt.Before(e.End) || tPnt.Equal(e.End)) {
				eventMask[i] = 1.0
			}
		}
		tFeat.Set(feat, eventMask)
	}

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

		// only model for daily since we're masking the weekends which means we do not meet the sampling requirements
		// to capture weekly seasonality.
		if opt.WeekendOptions.Enabled {
			tFeatName := "is_weekend"
			sFeatNamePrefix := "weekend_"

			eventSeasFeat, err := generateEventSeasonality(tFeat, dailyFeatures, tFeatName, sFeatNamePrefix)
			if err != nil {
				slog.Warn("unable to generate weekend daily seasonality", "time_feature_name", tFeatName)
			} else {
				x.Update(eventSeasFeat)
			}
		}

		for _, e := range opt.EventOptions.Events {
			tFeatName := fmt.Sprintf("event_%s", e.Name)
			sFeatNamePrefix := fmt.Sprintf("event_%s_" + e.Name)

			eventSeasFeat, err := generateEventSeasonality(tFeat, dailyFeatures, tFeatName, sFeatNamePrefix)
			if err != nil {
				slog.Warn("unable to generate event daily seasonality", "time_feature_name", tFeatName)
				continue
			}

			x.Update(eventSeasFeat)
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
		x.Update(weeklyFeatures)

		for _, e := range opt.EventOptions.Events {
			tFeatName := fmt.Sprintf("event_%s", e.Name)
			sFeatNamePrefix := fmt.Sprintf("event_%s_" + e.Name)

			eventSeasFeat, err := generateEventSeasonality(tFeat, weeklyFeatures, tFeatName, sFeatNamePrefix)
			if err != nil {
				slog.Warn("unable to generate event weekly seasonality", "time_feature_name", tFeatName)
				continue
			}

			x.Update(eventSeasFeat)
		}
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

func generateEventSeasonality(tFeat *feature.Set, sFeat *feature.Set, tFeatName string, sFeatNamePrefix string) (*feature.Set, error) {
	mask, exists := tFeat.Get(feature.NewTime(tFeatName))
	if !exists {
		return nil, fmt.Errorf("event mask not found, skipping event name, %s", tFeatName)
	}

	eventSeasonalityFeatures := feature.NewSet()
	for _, label := range sFeat.Labels() {
		featData, exists := sFeat.Get(label)
		if !exists {
			continue
		}
		maskedData := make([]float64, len(featData))
		floats.MulTo(maskedData, mask, featData)

		name, _ := label.Get("name")
		name = sFeatNamePrefix + name

		fcompStr, _ := label.Get("fourier_component")
		fcomp := feature.FourierComp(fcompStr)

		orderStr, _ := label.Get("order")
		order, _ := strconv.Atoi(orderStr)
		featCol := feature.NewSeasonality(name, fcomp, order)
		eventSeasonalityFeatures.Set(featCol, maskedData)
	}
	return eventSeasonalityFeatures, nil
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

func generateChangepointFeatures(t []time.Time, chpts []changepoint.Changepoint, trainingEndTime time.Time, enableGrowth bool) *feature.Set {
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
	chptBiasFeatures := make([][]float64, len(filteredChpts))
	var chptGrowthFeatures [][]float64
	if enableGrowth {
		chptGrowthFeatures = make([][]float64, len(filteredChpts))
	}
	for i := 0; i < len(filteredChpts); i++ {
		chptBias := make([]float64, len(t))
		chptBiasFeatures[i] = chptBias

		if enableGrowth {
			chptGrowth := make([]float64, len(t))
			chptGrowthFeatures[i] = chptGrowth
		}

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
				chptBiasFeatures[j][i] = bias

				if enableGrowth {
					slope = t[i].Sub(filteredChpts[j].T).Seconds() / deltaT[j]
					chptGrowthFeatures[j][i] = slope
				}
			}
		}
	}

	feat := feature.NewSet()
	for i := 0; i < len(chpts); i++ {
		chpntName := strconv.Itoa(i)
		if chpts[i].Name != "" {
			chpntName = chpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		feat.Set(chpntBias, chptBiasFeatures[i])

		if enableGrowth {
			chpntGrowth := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)
			feat.Set(chpntGrowth, chptGrowthFeatures[i])
		}
	}
	return feat
}

func loadLocationOffsets(names []string) []locDstOffset {
	var offsets []locDstOffset
	for _, name := range names {
		loc, err := time.LoadLocation(name)
		if err != nil {
			slog.Info("unable to load location, skipping", "location", name)
			continue
		}
		offset := getLocationDSTOffset(loc)
		offsets = append(offsets, locDstOffset{
			loc:    loc,
			offset: offset,
		})
	}
	return offsets
}

func getLocationDSTOffset(loc *time.Location) int {
	ctDec := time.Date(2024, 12, 1, 0, 0, 0, 0, loc)
	ctJul := time.Date(2024, 6, 1, 0, 0, 0, 0, loc)
	var offset int
	if ctDec.IsDST() {
		offset = deriveDSToffset(ctDec, ctJul)
	} else {
		offset = deriveDSToffset(ctJul, ctDec)
	}
	return offset
}

func deriveDSToffset(dstTime, stdTime time.Time) int {
	_, dstOffset := dstTime.Zone()
	_, stdOffset := stdTime.Zone()

	return dstOffset - stdOffset
}

type locDstOffset struct {
	loc    *time.Location
	offset int
}

// adjustTime checks a time against all dst offsets ranges and adjusts it by checking if the time is
// in a dst location offset range and finally averaging the cumulative offsets.
func adjustTime(t time.Time, offsets []locDstOffset) time.Time {
	var offsetSum int
	for _, offset := range offsets {
		locT := t.In(offset.loc)
		if locT.IsDST() {
			offsetSum += offset.offset
		}
	}
	if len(offsets) == 0 {
		return t
	}
	return t.Add(time.Duration(offsetSum/len(offsets)) * time.Second)
}

func adjustTimeForDST(t []time.Time, locNames []string) []time.Time {
	offsets := loadLocationOffsets(locNames)

	newT := make([]time.Time, len(t))
	for i := 0; i < len(t); i++ {
		newT[i] = adjustTime(t[i], offsets)
	}
	return newT
}
