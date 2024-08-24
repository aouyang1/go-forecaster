package main

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

var (
	ErrNoTrainingData           = errors.New("no training data")
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrUnknownTimeFeature       = errors.New("unknown time feature")
	ErrNoModelCoefficients      = errors.New("no model coefficients from fit")
)

type Forecast struct {
	trainingData *TimeDataset

	opt *Options

	tFeat map[string][]float64
	x     map[string][]float64

	coef []float64
}

func NewForecast(trainingData *TimeDataset, opt *Options) (*Forecast, error) {
	if trainingData == nil {
		return nil, ErrNoTrainingData
	}

	if opt == nil {
		opt = NewDefaultOptions()
	}

	forecast := &Forecast{
		trainingData: trainingData,
		opt:          opt,
		tFeat:        make(map[string][]float64),
		x:            make(map[string][]float64),
	}

	forecast.generateTimeFeatures()
	forecast.generateFourierFeatures()

	// prune linearly dependent fourier components

	return forecast, nil
}

func (f *Forecast) FeatureLabels() []string {
	labels := make([]string, 0, len(f.x))
	for label := range f.x {
		labels = append(labels, label)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i] < labels[j]
		},
	)
	return labels
}

func (f *Forecast) generateTimeFeatures() {
	if f.opt.DailyOrders > 0 {
		hod := make([]float64, len(f.trainingData.t))
		for i, tPnt := range f.trainingData.t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		f.tFeat["hod"] = hod
	}
	if f.opt.WeeklyOrders > 0 {
		dow := make([]float64, len(f.trainingData.t))
		for i, tPnt := range f.trainingData.t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		f.tFeat["dow"] = dow
	}
}

func (f *Forecast) generateFourierFeatures() {
	if f.opt.DailyOrders > 0 {
		f.generateFourierOrders("hod", f.opt.DailyOrders, 24.0)
	}

	if f.opt.WeeklyOrders > 0 {
		f.generateFourierOrders("dow", f.opt.WeeklyOrders, 7.0)
	}
}

func (f *Forecast) generateFourierOrders(tFeatCol string, orders int, period float64) error {
	tFeat, exists := f.tFeat[tFeatCol]
	if !exists {
		return fmt.Errorf("%s, %w", tFeatCol, ErrUnknownTimeFeature)
	}

	for order := 1; order <= orders; order++ {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := fmt.Sprintf("%s_%dsin", tFeatCol, order)
		cosFeatCol := fmt.Sprintf("%s_%dcos", tFeatCol, order)
		f.x[sinFeatCol] = sinFeat
		f.x[cosFeatCol] = cosFeat
	}
	return nil
}

func (f *Forecast) FeatureMatrix() mat.Matrix {
	m := len(f.trainingData.t)
	n := len(f.x) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < n; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	labels := f.FeatureLabels()
	for _, label := range labels {
		feature := f.x[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1

	}
	return mat.NewDense(m, n, obs)
}

func (f *Forecast) ObservationMatrix() mat.Matrix {
	n := len(f.trainingData.t)
	return mat.NewDense(1, n, f.trainingData.y)
}

func (f *Forecast) Fit() error {
	observations := f.ObservationMatrix()
	features := f.FeatureMatrix()
	f.coef = OLS(features, observations)
	return nil
}

func (f *Forecast) Coefficients() (map[string]float64, error) {
	labels := f.FeatureLabels()
	if len(labels) == 0 || len(f.coef) == 0 {
		return nil, ErrNoModelCoefficients
	}
	coef := make(map[string]float64)
	for i := 0; i < len(f.coef); i++ {
		if i == 0 {
			coef["bias"] = f.coef[0]
			continue
		}
		coef[labels[i-1]] = f.coef[i]
	}
	return coef, nil
}

func (f *Forecast) ModelEq() (string, error) {
	eq := "y ~ "

	coef, err := f.Coefficients()
	if err != nil {
		return "", err
	}

	labels := f.FeatureLabels()
	for i := 0; i < len(f.coef); i++ {
		if i == 0 {
			eq += fmt.Sprintf("%.2f", coef["bias"])
			continue
		}
		eq += fmt.Sprintf("+%.2f*%s", coef[labels[i-1]], labels[i-1])
	}
	return eq, nil
}
