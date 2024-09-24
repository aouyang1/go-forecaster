package feature

import (
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Set represents a mapping to each feature data keyed by the string representation
// of the feature.
type Set map[string]Data

// Labels returns the sorted slice of all tracked features in the FeatureSet
func (s Set) Labels() *Labels {
	if s == nil {
		return nil
	}

	labels := make([]Feature, 0, len(s))
	for _, feat := range s {
		labels = append(labels, feat.F)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i].String() < labels[j].String()
		},
	)
	return NewLabels(labels)
}

// Matrix returns a metric representation of the FeatureSet to be used with matrix methods
// The matrix has m rows representing the number of observations and n columns representing
// the number of features.
func (s Set) Matrix(intercept bool) *mat.Dense {
	if s == nil {
		return nil
	}

	featureLabels := s.Labels()
	if featureLabels.Len() == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels.Labels() {
		m = len(s[flabel.String()].Data)
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
		feature := s[label.String()]
		for i := 0; i < len(feature.Data); i++ {
			idx := n*i + featNum
			obs[idx] = feature.Data[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

// MatrixSlice returns the FeatureSet as a matrix but in the form of a slice of slices where
// each row represent feature. Takes an intercept input if we want to include the intercept
// term.
func (s Set) MatrixSlice(intercept bool) [][]float64 {
	if s == nil {
		return nil
	}

	featureLabels := s.Labels()
	if featureLabels.Len() == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels.Labels() {
		m = len(s[flabel.String()].Data)
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
		feature := s[label.String()]
		obs[featNum] = feature.Data
		featNum += 1
	}
	return obs
}
