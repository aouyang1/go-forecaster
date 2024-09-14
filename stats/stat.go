package stats

import (
	"errors"
	"math"
	"sort"

	"github.com/aouyang1/go-forecaster/models"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

var (
	ErrMinimumFeatures    = errors.New("need at least 2 features to compute VIF")
	ErrFeatureLenMismatch = errors.New("some feature length is not consistent")
	ErrFeatureLen         = errors.New("must have at least 2 points per feature")
)

func DetectOutliers(y []float64, lowerPerc, upperPerc, tukeyFactor float64) []int {
	lowerPerc = math.Max(lowerPerc, 0.0)
	upperPerc = math.Min(upperPerc, 1.0)
	tukeyFactor = math.Max(tukeyFactor, 0.0)

	yCopy := make([]float64, len(y))
	copy(yCopy, y)
	sort.Float64s(yCopy)
	lowerIdx := int(math.Floor(float64(len(yCopy)) * lowerPerc))
	upperIdx := int(math.Ceil(float64(len(yCopy)) * upperPerc))

	lower := yCopy[lowerIdx]
	upper := yCopy[upperIdx]
	innerRange := upper - lower
	lower -= innerRange * tukeyFactor
	upper += innerRange * tukeyFactor

	var outlierIdx []int
	for i := 0; i < len(y); i++ {
		if y[i] >= upper || y[i] <= lower {
			outlierIdx = append(outlierIdx, i)
		}
	}
	return outlierIdx
}

func VarianceInflationFactor(features map[string][]float64) (map[string]float64, error) {
	if len(features) < 2 {
		return nil, ErrMinimumFeatures
	}
	n := len(features)
	var m int
	for _, feature := range features {
		if len(feature) < 2 {
			return nil, ErrFeatureLen
		}
		if m == 0 {
			m = len(feature)
			continue
		}
		if m != len(feature) {
			return nil, ErrFeatureLenMismatch
		}
	}

	vif := make(map[string]float64)
	x := mat.NewDense(m, n, nil)
	y := mat.NewDense(1, m, nil)

	ones := make([]float64, m)
	floats.AddConst(1.0, ones)
	x.SetCol(0, ones)

	w := make([]float64, n)
	weightsMx := mat.NewDense(1, n, nil)
	for label, labelFeature := range features {
		y.SetRow(0, labelFeature)
		c := 1
		for otherLabel, otherLabelFeature := range features {
			if otherLabel == label {
				continue
			}
			x.SetCol(c, otherLabelFeature)
			c++
		}
		intercept, weights := models.OLS(x, y)
		w[0] = intercept
		for i, weight := range weights {
			w[i+1] = weight
		}
		weightsMx.SetRow(0, w)

		var predictedMx mat.Dense
		predictedMx.Mul(weightsMx, x.T())
		predicted := mat.Row(nil, 0, &predictedMx)

		vif[label] = stat.RSquaredFrom(predicted, labelFeature, nil)
	}
	return vif, nil
}
