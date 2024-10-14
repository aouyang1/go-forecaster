package feature

import (
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Set represents a mapping to each feature data keyed by the string representation
// of the feature.
type Set struct {
	set map[string]Data

	labels []Feature
}

func NewSet() *Set {
	return &Set{
		set: make(map[string]Data),
	}
}

func (s *Set) Len() int {
	return len(s.set)
}

func (s *Set) Get(f Feature) (Data, bool) {
	label := f.String()
	data, exists := s.set[label]
	return data, exists
}

func (s *Set) Set(f Feature, data Data) {
	label := f.String()
	if _, exists := s.set[label]; !exists {
		s.labels = append(s.labels, f)
	}
	s.set[label] = data
}

func (s *Set) Del(f Feature) {
	label := f.String()
	if _, exists := s.set[label]; !exists {
		return
	}
	delete(s.set, label)
	for i, l := range s.labels {
		if l.String() == f.String() {
			temp := s.labels[0]
			s.labels[0] = l
			s.labels[i] = temp
			break
		}
	}
	s.labels = s.labels[1:]
}

func (s *Set) Update(other *Set) {
	if other == nil {
		return
	}
	if other.set == nil {
		return
	}
	for _, f := range other.labels {
		label := f.String()
		if _, exists := s.set[label]; !exists {
			s.labels = append(s.labels, f)
		}
		s.set[label] = other.set[label]
	}
}

// Labels returns the sorted slice of all tracked features in the FeatureSet
func (s *Set) Labels() []Feature {
	if s == nil {
		return nil
	}

	labels := make([]Feature, len(s.labels))
	copy(labels, s.labels)
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i].String() < labels[j].String()
		},
	)
	return labels
}

// Matrix returns a metric representation of the FeatureSet to be used with matrix methods
// The matrix has m rows representing the number of observations and n columns representing
// the number of features.
func (s *Set) Matrix(intercept bool) *mat.Dense {
	if s == nil {
		return nil
	}

	featureLabels := s.Labels()
	if len(featureLabels) == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels {
		m = len(s.set[flabel.String()].Data)
		break
	}
	n := len(featureLabels)
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

	for _, label := range featureLabels {
		feature := s.set[label.String()]
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
func (s *Set) MatrixSlice(intercept bool) [][]float64 {
	if s == nil {
		return nil
	}

	featureLabels := s.Labels()
	if len(featureLabels) == 0 {
		return nil
	}

	var m int
	// use first feature to get length
	for _, flabel := range featureLabels {
		m = len(s.set[flabel.String()].Data)
		break
	}
	n := len(featureLabels)
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

	for _, label := range featureLabels {
		feature := s.set[label.String()]
		obs[featNum] = feature.Data
		featNum += 1
	}
	return obs
}
