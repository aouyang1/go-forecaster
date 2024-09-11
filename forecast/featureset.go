package forecast

import (
	"sort"

	"github.com/aouyang1/go-forecast/feature"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type Feature struct {
	F    feature.Feature
	Data []float64
}
type FeatureSet map[string]Feature

func (f FeatureSet) Labels() *FeatureLabels {
	if f == nil {
		return nil
	}

	labels := make([]feature.Feature, 0, len(f))
	for _, feat := range f {
		labels = append(labels, feat.F)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i].String() < labels[j].String()
		},
	)
	return NewFeatureLabels(labels)
}

func (f FeatureSet) Matrix(intercept bool) *mat.Dense {
	if f == nil {
		return nil
	}

	featureLabels := f.Labels()
	if featureLabels.Len() == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels.Labels() {
		m = len(f[flabel.String()].Data)
		break
	}
	n := featureLabels.Len()
	if intercept {
		n += 1
	}

	obs := make([]float64, m*n)

	featNum := 0
	if intercept {
		for i := 0; i < m; i++ {
			idx := n * i
			obs[idx] = 1.0
		}
		featNum += 1
	}

	for _, label := range featureLabels.Labels() {
		feature := f[label.String()]
		for i := 0; i < len(feature.Data); i++ {
			idx := n*i + featNum
			obs[idx] = feature.Data[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

func (f FeatureSet) MatrixSlice(intercept bool) [][]float64 {
	if f == nil {
		return nil
	}

	featureLabels := f.Labels()
	if featureLabels.Len() == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels.Labels() {
		m = len(f[flabel.String()].Data)
		break
	}
	n := featureLabels.Len()
	if intercept {
		n += 1
	}

	obs := make([][]float64, n)
	featNum := 0
	if intercept {
		ones := make([]float64, m)
		floats.AddConst(1.0, ones)
		obs[featNum] = ones
		featNum++
	}

	for _, label := range featureLabels.Labels() {
		feature := f[label.String()]
		obs[featNum] = feature.Data
		featNum += 1
	}
	return obs
}
