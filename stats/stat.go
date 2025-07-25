// Package stats is a collection of helpful statistical methods
package stats

import (
	"math"
	"sort"
)

// DetectOutliers uses the Tukey Method to return a slice of indexes that are classified as outliers
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
	for i := range len(y) {
		if y[i] > upper || y[i] < lower {
			outlierIdx = append(outlierIdx, i)
		}
	}
	return outlierIdx
}
