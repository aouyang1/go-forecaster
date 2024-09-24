package feature

// Labels tracks a slice of features and their index locations that match up
// with the ordering of the coefficients assigned to each of these features.
type Labels struct {
	idx    map[string]int
	labels []Feature
}

func NewLabels(labels []Feature) *Labels {
	idx := make(map[string]int)
	for i := 0; i < len(labels); i++ {
		idx[labels[i].String()] = i
	}
	fl := &Labels{
		labels: labels,
		idx:    idx,
	}
	return fl
}

func (f *Labels) Len() int {
	return len(f.labels)
}

func (f *Labels) Labels() []Feature {
	labels := make([]Feature, len(f.labels))
	copy(labels, f.labels)
	return labels
}

func (f *Labels) Index(label Feature) (int, bool) {
	if idx, exists := f.idx[label.String()]; exists {
		return idx, exists
	}
	return -1, false
}
