package forecast

import "github.com/aouyang1/go-forecaster/feature"

// FeatureData represents a feature type with is associated observed values
type FeatureData struct {
	F    feature.Feature
	Data []float64
}
