package feature

import (
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Set represents a mapping to each feature data keyed by the string representation
// of the feature.
type Set struct {
	set map[string][]float64

	labels []Feature
}

func NewSet() *Set {
	return &Set{
		set: make(map[string][]float64),
	}
}

func (s *Set) Len() int {
	return len(s.set)
}

func (s *Set) Get(f Feature) ([]float64, bool) {
	label := f.String()
	data, exists := s.set[label]
	return data, exists
}

func (s *Set) Set(f Feature, data []float64) *Set {
	label := f.String()
	if _, exists := s.set[label]; !exists {
		s.labels = append(s.labels, f)
	}
	s.set[label] = data
	return s
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
		m = len(s.set[flabel.String()])
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
		for i, pnt := range s.set[label.String()] {
			idx := n*i + featNum
			obs[idx] = pnt

		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}

func (s *Set) RemoveZeroOnlyFeatures() {
	for _, feat := range s.Labels() {
		vals, _ := s.Get(feat)
		dot := floats.Dot(vals, vals)
		if dot == 0 {
			s.Del(feat)
		}
	}
}
