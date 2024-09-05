package forecast

import (
	"sort"

	"github.com/aouyang1/go-forecast/feature"
	"gonum.org/v1/gonum/mat"
)

type FeatureSet map[feature.Feature][]float64

func (f FeatureSet) Labels() []feature.Feature {
	if f == nil {
		return nil
	}

	labels := make([]feature.Feature, 0, len(f))
	for label := range f {
		labels = append(labels, label)
	}
	sort.Slice(
		labels,
		func(i, j int) bool {
			return labels[i].String() < labels[j].String()
		},
	)
	return labels
}

func (f FeatureSet) Matrix() *mat.Dense {
	if f == nil {
		return nil
	}

	featureLabels := f.Labels()
	if len(featureLabels) == 0 {
		return nil
	}

	m := len(f[featureLabels[0]])
	n := len(featureLabels) + 1
	obs := make([]float64, m*n)

	featNum := 0
	for i := 0; i < m; i++ {
		idx := n * i
		obs[idx] = 1.0
	}
	featNum += 1

	for _, label := range featureLabels {
		feature := f[label]
		for i := 0; i < len(feature); i++ {
			idx := n*i + featNum
			obs[idx] = feature[i]
		}
		featNum += 1
	}
	return mat.NewDense(m, n, obs)
}
