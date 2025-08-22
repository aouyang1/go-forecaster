package feature

import (
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Set represents a mapping to each feature data keyed by the string representation
// of the feature.
type Set struct {
	m   int // length of the dataset by key
	set map[string][]float64

	labels []Feature
}

// NewSet initializes a new feature set
func NewSet() *Set {
	return &Set{
		set:    make(map[string][]float64),
		labels: []Feature{},
	}
}

// Len returns the number of labels stored in the feature set
func (s *Set) Len() int {
	return len(s.set)
}

// Get retrieves the values of a particular feature. This returns an extra bool to indicate
// if the feature exists in the set or not.
func (s *Set) Get(f Feature) ([]float64, bool) {
	label := f.String()
	data, exists := s.set[label]
	return data, exists
}

// Set will store the input feature with the input slice of values. This will zero pad out
// the data if it is less than the current data length of the set. If it is larger, all other
// features will be zero padded.
func (s *Set) Set(f Feature, data []float64) *Set {
	label := f.String()
	if _, exists := s.set[label]; !exists {
		s.labels = append(s.labels, f)
	}

	// input data is longer than current feature set length
	if s.m < len(data) {
		s.m = len(data)
		for label, vals := range s.set {
			nextVals := make([]float64, s.m)
			copy(nextVals, vals)
			s.set[label] = nextVals
		}
		s.set[label] = data
		return s
	}

	// data length is less than current length of feature set
	if s.m > len(data) {
		nextVals := make([]float64, s.m)
		copy(nextVals, data)
		s.set[label] = nextVals
		return s
	}

	s.set[label] = data

	return s
}

// Del will attempt to remove an existing feature from the set. If there are no more features
// stored, the data length will be reset to 0 and will be initialized on the next Set call.
func (s *Set) Del(f Feature) *Set {
	label := f.String()
	if _, exists := s.set[label]; !exists {
		return s
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

	if s.Len() == 0 {
		s.m = 0
	}
	return s
}

// Update will merge an input set with the existing set. If the data length of the input is less
// than the existing, the input is zero padded out to match. If the data length of the input is greater
// than the existing, the existing is zero padded out to match. Any overlapping keys will result in
// a replacement.
func (s *Set) Update(other *Set) *Set {
	if other == nil {
		return s
	}
	if other.set == nil {
		return s
	}
	if s.m < other.m {
		s.m = other.m
		// buffer existing set
		for label, vals := range s.set {
			nextVals := make([]float64, s.m)
			copy(nextVals, vals)
			s.set[label] = nextVals
		}
	} else if s.m > other.m {
		// buffer data
		for label, vals := range other.set {
			nextVals := make([]float64, s.m)
			copy(nextVals, vals)
			other.set[label] = nextVals
		}
	}
	for _, f := range other.labels {
		label := f.String()
		if _, exists := s.set[label]; !exists {
			s.labels = append(s.labels, f)
		}
		s.set[label] = other.set[label]
	}
	return s
}

// Labels returns the sorted slice of all tracked features in the FeatureSet prioritizing
// intercept to be the first
func (s *Set) Labels() []Feature {
	if s == nil {
		return nil
	}
	interceptFeat := Intercept()

	labels := make([]Feature, len(s.labels))
	copy(labels, s.labels)
	sort.Slice(
		labels,
		func(i, j int) bool {
			if labels[i].String() == interceptFeat.String() {
				return true
			}
			if labels[j].String() == interceptFeat.String() {
				return false
			}

			return labels[i].String() < labels[j].String()
		},
	)
	return labels
}

// Matrix returns a metric representation of the FeatureSet to be used with matrix methods
// The matrix has m rows representing the number of observations and n columns representing
// the number of features.
func (s *Set) Matrix() *mat.Dense {
	if s == nil {
		return nil
	}

	featureLabels := s.Labels()
	if len(featureLabels) == 0 {
		return nil
	}

	m := s.m
	n := len(featureLabels)

	obs := make([]float64, m*n)

	for featNum, label := range featureLabels {
		for i, pnt := range s.set[label.String()] {
			idx := n*i + featNum
			obs[idx] = pnt
		}
	}
	return mat.NewDense(m, n, obs)
}

// RemoveZeroOnlyFeatures scans through all features and removes any features with only zero values.
// This is to prevent fitting issues.
func (s *Set) RemoveZeroOnlyFeatures() {
	for _, feat := range s.Labels() {
		vals, _ := s.Get(feat)
		dot := floats.Dot(vals, vals)
		if dot == 0 {
			s.Del(feat)
		}
	}
}
