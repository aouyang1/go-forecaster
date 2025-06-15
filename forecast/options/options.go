package options

import (
	"errors"
	"fmt"
	"log/slog"
	"math"
	"strconv"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/models"
	"gonum.org/v1/gonum/floats"
)

const (
	LabelTimeEpoch = "epoch"

	LabelSeasDaily  = "daily"
	LabelSeasWeekly = "weekly"

	LabelEventWeekend = "weekend"
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

	DSTOptions     DSTOptions     `json:"dst_options"`
	WeekendOptions WeekendOptions `json:"weekend_options"`
	EventOptions   EventOptions   `json:"event_options"`
	MaskWindow     string         `json:"mask_window"`
}

// NewDefaultOptions returns a set of default forecast options
func NewDefaultOptions() *Options {
	return &Options{
		ChangepointOptions: NewDefaultChangepointOptions(),
		Regularization:     []float64{0.0},
		SeasonalityOptions: NewDefaultSeasonalityOptions(),
	}
}

func (o *Options) NewLassoAutoOptions() *models.LassoAutoOptions {
	lassoOpt := models.NewDefaultLassoAutoOptions()
	if len(o.Regularization) > 0 {
		lassoOpt.Lambdas = o.Regularization
	} else {
		o.Regularization = lassoOpt.Lambdas
	}

	lassoOpt.FitIntercept = false

	lassoOpt.Iterations = o.Iterations
	if o.Iterations == 0 {
		lassoOpt.Iterations = models.DefaultIterations
	}

	lassoOpt.Tolerance = o.Tolerance
	if o.Tolerance == 0 {
		lassoOpt.Tolerance = models.DefaultTolerance
	}

	lassoOpt.Parallelization = o.Parallelization
	return lassoOpt
}

func (o *Options) GenerateTimeFeatures(t []time.Time) (*feature.Set, *feature.Set) {
	if o == nil {
		o = NewDefaultOptions()
	}

	tFeat := feature.NewSet()

	interceptFeat := feature.Intercept()
	ones := make([]float64, len(t))
	floats.AddConst(1.0, ones)
	tFeat.Set(interceptFeat, ones)

	epoch := make([]float64, len(t))
	for i, tPnt := range t {
		epochNano := float64(tPnt.UnixNano()) / 1e9
		epoch[i] = epochNano
	}
	feat := feature.NewTime(LabelTimeEpoch)
	tFeat.Set(feat, epoch)

	eFeat := o.GenerateEventFeatures(t)
	tFeat.Update(eFeat)

	return tFeat, eFeat
}

func (o *Options) GenerateEventFeatures(t []time.Time) *feature.Set {
	if o == nil {
		o = NewDefaultOptions()
	}

	winFunc := WindowFunc(o.MaskWindow)

	eFeat := feature.NewSet()

	o.WeekendOptions.generateEventMask(t, eFeat, winFunc)
	o.EventOptions.generateEventMask(t, eFeat, winFunc)
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

		if seasCfg.Name == LabelSeasDaily && o.WeekendOptions.Enabled {
			// only model for daily since we're masking the weekends which means we do not meet the sampling requirements
			// to capture weekly seasonality.
			eventSeasFeat, err := generateEventSeasonality(feat, seasFeatures, LabelEventWeekend, LabelSeasDaily)
			if err != nil {
				slog.Warn("unable to generate weekend daily seasonality", "feature_name", LabelEventWeekend, "error", err.Error())
			} else {
				x.Update(eventSeasFeat)
			}
		}

		// generate seasonality features for events
		for _, e := range o.EventOptions.Events {
			eventSeasFeat, err := generateEventSeasonality(feat, seasFeatures, e.Name, seasCfg.Name)
			if err != nil {
				slog.Warn("unable to generate event seasonality", "feature_name", e.Name, "seasonality", seasCfg.Name, "error", err.Error())
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
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := feature.NewSeasonality(col+"_"+label, feature.FourierCompSin, order)
		cosFeatCol := feature.NewSeasonality(col+"_"+label, feature.FourierCompCos, order)
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

func generateEventSeasonality(feat, sFeat *feature.Set, eCol, sLabel string) (*feature.Set, error) {
	mask, exists := feat.Get(feature.NewEvent(eCol))
	if !exists {
		return nil, fmt.Errorf("event mask not found, skipping event name, %s", eCol)
	}

	eventSeasonalityFeatures := feature.NewSet()
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
		featCol := feature.NewSeasonality(eCol+"_"+sLabel, fcomp, order)
		eventSeasonalityFeatures.Set(featCol, maskedData)
	}
	return eventSeasonalityFeatures, nil
}
