package main

import (
	"fmt"
	"math"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

func featureMatrix(t []time.Time, featureLabels []string, features map[string][]float64) mat.Matrix {
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

func observationMatrix(y []float64) mat.Matrix {
	n := len(y)
	return mat.NewDense(1, n, y)
}

func generateTimeFeatures(t []time.Time, opt *Options) map[string][]float64 {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	tFeat := make(map[string][]float64)
	if opt.DailyOrders > 0 {
		hod := make([]float64, len(t))
		for i, tPnt := range t {
			hour := float64(tPnt.Unix()) / 3600.0
			hod[i] = math.Mod(hour, 24.0)
		}
		tFeat["hod"] = hod
	}
	if opt.WeeklyOrders > 0 {
		dow := make([]float64, len(t))
		for i, tPnt := range t {
			day := float64(tPnt.Unix()) / 86400.0
			dow[i] = math.Mod(day, 7.0)
		}
		tFeat["dow"] = dow
	}
	return tFeat
}

func featureLabels(features map[string][]float64) []string {
	labels := make([]string, 0, len(features))
	for label := range features {
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

func generateFourierFeatures(tFeat map[string][]float64, opt *Options) (map[string][]float64, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}
	x := make(map[string][]float64)
	if opt.DailyOrders > 0 {
		dailyFeatures, err := generateFourierOrders(tFeat, "hod", opt.DailyOrders, 24.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "hod", err)
		}
		for label, features := range dailyFeatures {
			x[label] = features
		}
	}

	if opt.WeeklyOrders > 0 {
		weeklyFeatures, err := generateFourierOrders(tFeat, "dow", opt.WeeklyOrders, 7.0)
		if err != nil {
			return nil, fmt.Errorf("%q not present in time features, %w", "dow", err)
		}
		for label, features := range weeklyFeatures {
			x[label] = features
		}
	}
	return x, nil
}

func generateFourierOrders(tFeatures map[string][]float64, col string, orders int, period float64) (map[string][]float64, error) {
	tFeat, exists := tFeatures[col]
	if !exists {
		return nil, ErrUnknownTimeFeature
	}

	x := make(map[string][]float64)
	for order := 1; order <= orders; order++ {
		sinFeat, cosFeat := generateFourierComponent(tFeat, order, period)
		sinFeatCol := fmt.Sprintf("%s_%dsin", col, order)
		cosFeatCol := fmt.Sprintf("%s_%dcos", col, order)
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

func OLS(obs, y mat.Matrix) (float64, []float64) {
	_, n := obs.Dims()
	qr := new(mat.QR)
	qr.Factorize(obs)

	q := new(mat.Dense)
	r := new(mat.Dense)

	qr.QTo(q)
	qr.RTo(r)
	yq := new(mat.Dense)
	yq.Mul(y, q)

	c := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		c[i] = yq.At(0, i)
		for j := i + 1; j < n; j++ {
			c[i] -= c[j] * r.At(i, j)
		}
		c[i] /= r.At(i, i)
	}
	if len(c) == 0 {
		return math.NaN(), nil
	}
	if len(c) == 1 {
		return c[0], nil
	}
	return c[0], c[1:]
}
