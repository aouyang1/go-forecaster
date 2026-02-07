// Package options contains all forecast options for a linear fit of a univariate time series
package options

import (
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/linearmodel"
	"gonum.org/v1/gonum/floats"
)

const (
	LabelTimeEpoch = "epoch"

	LabelSeasDaily  = "daily"
	LabelSeasWeekly = "weekly"
)

var ErrUnknownTimeFeature = errors.New("unknown time feature")

// Options configures a forecast by specifying changepoints, seasonality order
// and an optional regularization parameter where higher values removes more features
// that contribute the least to the fit.
type Options struct {
	UseLog bool `json:"use_log"`

	ChangepointOptions ChangepointOptions `json:"changepoint_options"`

	// Lasso related options
	Regularization  []float64 `json:"regularization"`
	Iterations      int       `json:"iterations"`
	Tolerance       float64   `json:"tolerance"`
	Parallelization int       `json:"parallelization"`

	SeasonalityOptions SeasonalityOptions `json:"seasonality_options"`

	DSTOptions         DSTOptions         `json:"dst_options"`
	WeekendOptions     WeekendOptions     `json:"weekend_options"`
	EventSeriesOptions EventSeriesOptions `json:"-"`
	EventOptions       EventOptions       `json:"event_options"`
	MaskWindow         string             `json:"mask_window"`
	GrowthType         string             `json:"growth_type"`
}

// NewDefaultOptions returns a set of default forecast options
func NewDefaultOptions() *Options {
	return &Options{
		ChangepointOptions: NewDefaultChangepointOptions(),
		Regularization:     []float64{0.0},
		SeasonalityOptions: NewDefaultSeasonalityOptions(),
	}
}

func (o *Options) NewLassoAutoOptions() *linearmodel.LassoAutoOptions {
	lassoOpt := linearmodel.NewDefaultLassoAutoOptions()
	if len(o.Regularization) > 0 {
		lassoOpt.Lambdas = o.Regularization
	} else {
		o.Regularization = lassoOpt.Lambdas
	}

	lassoOpt.FitIntercept = false

	lassoOpt.Iterations = o.Iterations
	if o.Iterations == 0 {
		lassoOpt.Iterations = linearmodel.DefaultIterations
	}

	lassoOpt.Tolerance = o.Tolerance
	if o.Tolerance == 0 {
		lassoOpt.Tolerance = linearmodel.DefaultTolerance
	}

	lassoOpt.Parallelization = o.Parallelization
	return lassoOpt
}

func (o *Options) GenerateTimeFeatures(t []time.Time, trainStartTime, trainEndTime time.Time) (*feature.Set, *feature.Set) {
	if o == nil {
		o = NewDefaultOptions()
	}

	tFeat := feature.NewSet()

	feat := feature.NewTime(LabelTimeEpoch)
	epoch := feat.Generate(t)
	tFeat.Set(feat, epoch)

	o.generateGrowthFeatures(epoch, trainStartTime, trainEndTime, tFeat)

	eFeat := o.GenerateEventFeatures(t)
	tFeat.Update(eFeat)

	return tFeat, eFeat
}

func (o *Options) generateGrowthFeatures(epoch []float64, trainStartTime, trainEndTime time.Time, tFeat *feature.Set) {
	if trainEndTime.Equal(trainStartTime) {
		return
	}
	interceptFeat := feature.Intercept()
	tFeat.Set(interceptFeat, interceptFeat.Generate(epoch, trainStartTime, trainEndTime))

	if o.GrowthType == "" {
		return
	}

	// Add growth features if specified
	switch o.GrowthType {
	case feature.GrowthLinear:
		linearFeat := feature.Linear()
		tFeat.Set(linearFeat, linearFeat.Generate(epoch, trainStartTime, trainEndTime))

	case feature.GrowthQuadratic:
		quadraticFeat := feature.Quadratic()
		tFeat.Set(quadraticFeat, quadraticFeat.Generate(epoch, trainStartTime, trainEndTime))
	}
}

func (o *Options) GenerateEventFeatures(t []time.Time) *feature.Set {
	if o == nil {
		o = NewDefaultOptions()
	}

	eFeat := feature.NewSet()

	eventSeriesOpts := o.EventSeriesOptions
	if o.WeekendOptions.Enabled {
		eventSeriesOpts = append(eventSeriesOpts, &o.WeekendOptions)
	}

	for eventSeriesOpt := range slices.Values(eventSeriesOpts) {
		mask, err := eventSeriesOpt.GenerateMask(t, o.MaskWindow)
		if err != nil {
			slog.Warn("unable to generate mask", "name", eventSeriesOpt.Name())
		}
		if err := RegisterEventSeries(eventSeriesOpt.Name(), eFeat, mask); err != nil {
			slog.Warn("unable to register event series mask", "name", eventSeriesOpt.Name(), "error", err.Error())
		}
	}

	o.EventOptions.generateEventMask(t, eFeat, o.MaskWindow)
	return eFeat
}

func (o *Options) GenerateFourierFeatures(feat *feature.Set) (*feature.Set, error) {
	if o == nil {
		o = NewDefaultOptions()
	}
	x := feature.NewSet()

	o.SeasonalityOptions.removeDuplicates()
	colinearCfgOrders := o.SeasonalityOptions.colinearConfigOrders()

	for _, seasCfg := range o.SeasonalityOptions.SeasonalityConfigs {
		orders := seasCfg.filterOutColinearOrders(colinearCfgOrders)
		seasFeatures, err := generateFourierOrders(feat, orders, seasCfg.Period, seasCfg.Name)
		if err != nil {
			return nil, fmt.Errorf("unable to generate seasonality features for %q, %w", seasCfg.Name, err)
		}
		x.Update(seasFeatures)

		// generate seasonality features for events
		for _, e := range o.EventOptions.Events {
			eventSeasFeat, err := generateEventSeasonality(feat, seasFeatures, e.Name, seasCfg.Name)
			if err != nil {
				slog.Warn("unable to generate event seasonality", "feature_name", e.Name, "seasonality", seasCfg.Name, "error", err.Error())
				continue
			}

			x.Update(eventSeasFeat)
		}

		// generate seasonality features for changepoints
		for _, c := range o.ChangepointOptions.Changepoints {
			changepointSeasFeat, err := generateChangepointSeasonality(feat, seasFeatures, c.Name, seasCfg.Name)
			if err != nil {
				slog.Warn("unable to generate changepoint seasonality", "feature_name", c.Name, "seasonality", seasCfg.Name, "error", err.Error())
				continue
			}

			x.Update(changepointSeasFeat)
		}

		// weekend seasonality
		if seasCfg.Name == LabelSeasDaily && o.WeekendOptions.Enabled {
			// only model for daily since we're masking the weekends which means we do not meet the sampling requirements
			// to capture weekly seasonality.
			eventSeasFeat, err := generateEventSeasonality(feat, seasFeatures, LabelEventWeekend, LabelSeasDaily)
			if err != nil {
				slog.Warn("unable to generate weekend daily seasonality", "feature_name", LabelEventWeekend, "error", err.Error())
				continue
			}
			x.Update(eventSeasFeat)
		}

	}
	return x, nil
}

func generateFourierOrders(tFeatures *feature.Set, orders []int, periodDur time.Duration, label string) (*feature.Set, error) {
	if tFeatures == nil {
		return nil, ErrUnknownTimeFeature
	}

	col := LabelTimeEpoch
	tFeat, exists := tFeatures.Get(feature.NewTime(col))
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	period := periodDur.Seconds()

	x := feature.NewSet()
	for _, order := range orders {
		sinFeat := feature.NewSeasonality(col+"_"+label, feature.FourierCompSin, order)
		cosFeat := feature.NewSeasonality(col+"_"+label, feature.FourierCompCos, order)
		x.Set(sinFeat, sinFeat.Generate(tFeat, order, period))
		x.Set(cosFeat, cosFeat.Generate(tFeat, order, period))
	}

	return x, nil
}

func generateEventSeasonality(feat, sFeat *feature.Set, eCol, sLabel string) (*feature.Set, error) {
	mask, exists := feat.Get(feature.NewEvent(eCol))
	if !exists {
		return nil, fmt.Errorf("feature event mask not found, skipping event feature name, %s", eCol)
	}

	return generateMaskedSeasonality(sFeat, eCol, mask, sLabel), nil
}

func generateChangepointSeasonality(feat, sFeat *feature.Set, cCol, sLabel string) (*feature.Set, error) {
	mask, exists := feat.Get(feature.NewChangepoint(cCol, feature.ChangepointCompBias))
	if !exists {
		return nil, fmt.Errorf("feature changepoint mask not found, skipping changepoint feature name, %s", cCol)
	}

	return generateMaskedSeasonality(sFeat, cCol, mask, sLabel), nil
}

func generateMaskedSeasonality(sFeat *feature.Set, col string, mask []float64, sLabel string) *feature.Set {
	maskedSeasonalityFeatures := feature.NewSet()
	for _, label := range sFeat.Labels() {
		featData, exists := sFeat.Get(label)
		if !exists {
			continue
		}
		maskedData := make([]float64, len(featData))
		floats.MulTo(maskedData, mask, featData)

		fcompStr, _ := label.Get("fourier_component")
		fcomp := feature.FourierComp(fcompStr)

		orderStr, _ := label.Get("order")
		order, _ := strconv.Atoi(orderStr)
		featCol := feature.NewSeasonality(col+"_"+sLabel, fcomp, order)
		maskedSeasonalityFeatures.Set(featCol, maskedData)
	}
	return maskedSeasonalityFeatures
}
