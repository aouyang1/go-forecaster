package main

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

var (
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrUnknownTimeFeature       = errors.New("unknown time feature")
)

type Options struct {
	DailyOrders  int
	WeeklyOrders int
}

func NewDefaultOptions() *Options {
	return &Options{
		DailyOrders:  12,
		WeeklyOrders: 12,
	}
}

type TimeDataset struct {
	opt *Options

	t     []time.Time
	tFeat map[string][]float64

	x map[string][]float64
	y []float64
}

func NewUnivariateDataset(t []time.Time, y []float64, opt *Options) (*TimeDataset, error) {
	if len(t) != len(y) {
		return nil, fmt.Errorf("time feature has length of %d, but values has a length of %d", len(t), len(y))
	}

	if opt == nil {
		opt = NewDefaultOptions()
	}

	td := &TimeDataset{
		opt: opt,

		t:     t,
		tFeat: make(map[string][]float64),

		x: make(map[string][]float64),
		y: y,
	}

	td.generateTimeFeatures()
	td.generateFourierFeatures()

	// prune linearly dependent fourier components
	return td, nil
}

func (td *TimeDataset) FeatureLabels() []string {
	labels := make([]string, 0, len(td.x))
	for label := range td.x {
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

func (td *TimeDataset) generateTimeFeatures() {
	if td.opt.DailyOrders > 0 {
		hod := make([]float64, len(td.t))
		for i, tPnt := range td.t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		td.tFeat["hod"] = hod
	}
	if td.opt.WeeklyOrders > 0 {
		dow := make([]float64, len(td.t))
		for i, tPnt := range td.t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		td.tFeat["dow"] = dow
	}
}

func (td *TimeDataset) generateFourierFeatures() {
	if td.opt.DailyOrders > 0 {
		td.generateFourierOrders("hod", td.opt.DailyOrders, 24.0)
	}

	if td.opt.WeeklyOrders > 0 {
		td.generateFourierOrders("dow", td.opt.WeeklyOrders, 7.0)
	}
}

func (td *TimeDataset) generateFourierOrders(tFeatCol string, orders int, period float64) error {
	tFeat, exists := td.tFeat[tFeatCol]
	if !exists {
		return fmt.Errorf("%s, %w", tFeatCol, ErrUnknownTimeFeature)
	}

	for order := 1; order <= orders; order++ {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := fmt.Sprintf("%s_%dsin", tFeatCol, order)
		cosFeatCol := fmt.Sprintf("%s_%dcos", tFeatCol, order)
		td.x[sinFeatCol] = sinFeat
		td.x[cosFeatCol] = cosFeat
	}
	return nil
}

func (td *TimeDataset) FeatureMatrix() mat.Matrix {
	m := len(td.t)
	n := len(td.x) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < n; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	labels := td.FeatureLabels()
	for _, label := range labels {
		feature := td.x[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1

	}
	return mat.NewDense(m, n, obs)
}

func (td *TimeDataset) ObservationMatrix() mat.Matrix {
	n := len(td.t)
	return mat.NewDense(1, n, td.y)
}

func main() {
	// create a daily sine wave at minutely with one week
	minutes := 7 * 24 * 60
	// minutes = 60
	t := make([]time.Time, 0, minutes)
	ct := time.Now().Add(-time.Duration(6) * time.Hour)
	for i := 0; i < minutes; i++ {
		t = append(t, ct.Add(-time.Duration(i)*time.Minute))
	}
	y := make([]float64, 0, minutes)
	for i := 0; i < minutes; i++ {
		y = append(y, 1.2+4.3*math.Sin(2.0*math.Pi/86400.0*float64(t[i].Unix()+3*60*60)))
	}
	opt := &Options{
		DailyOrders: 3,
	}
	td, err := NewUnivariateDataset(t, y, opt)
	if err != nil {
		panic(err)
	}
	observations := td.ObservationMatrix()
	features := td.FeatureMatrix()
	coef := OLS(features, observations)

	labels := td.FeatureLabels()
	for i := 0; i < len(coef); i++ {
		if i == 0 {
			fmt.Printf("bias: %.4f\n", coef[i])
			continue
		}
		fmt.Printf("%s: %.4f\n", labels[i-1], coef[i])
	}
}
