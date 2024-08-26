package main

import (
	"math"
	"sort"
)

func DetectOutliers(opt *OutlierOptions, y []float64) []int {
	if opt == nil {
		return nil
	}
	opt.LowerPercentile = math.Max(opt.LowerPercentile, 0.0)
	opt.UpperPercentile = math.Min(opt.UpperPercentile, 1.0)

	yCopy := make([]float64, len(y))
	copy(yCopy, y)
	sort.Float64s(yCopy)
	lowerIdx := int(math.Floor(float64(len(yCopy)) * opt.LowerPercentile))
	upperIdx := int(math.Ceil(float64(len(yCopy)) * opt.UpperPercentile))

	lower := yCopy[lowerIdx]
	upper := yCopy[upperIdx]
	innerRange := upper - lower
	lower -= innerRange * opt.TukeyFactor
	upper += innerRange * opt.TukeyFactor

	var outlierIdx []int
	for i := 0; i < len(y); i++ {
		if y[i] >= upper || y[i] <= lower {
			outlierIdx = append(outlierIdx, i)
		}
	}
	return outlierIdx
}
