package feature

type FeatureType string

const (
	FeatureTypeGrowth      FeatureType = "growth"
	FeatureTypeChangepoint FeatureType = "changepoint"
	FeatureTypeSeasonality FeatureType = "seasonality"
	FeatureTypeTime        FeatureType = "time"
	FeatureTypeEvent       FeatureType = "event"
)

// Feature is an interface representing a type of feature e.g. changepoint,
// seasonality, or time
type Feature interface {
	// String returns the string representation of the feature
	String() string

	// Get returns the value of an arbitrary label and returns the value along with whether
	// the label exists
	Get(string) (string, bool)

	// Type returns the type of this feature
	Type() FeatureType

	// Decode converts the feature into a map of label values
	Decode() map[string]string

	// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
	// to a feature
	UnmarshalJSON([]byte) error
}
