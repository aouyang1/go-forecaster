package feature

type FeatureType string

const (
	FeatureTypeChangepoint FeatureType = "changepoint"
	FeatureTypeSeasonality FeatureType = "seasonality"
	FeatureTypeTime        FeatureType = "time"
)

type Feature interface {
	String() string
	Get(string) (string, bool)
	Type() FeatureType
	Decode() map[string]string
	UnmarshalJSON([]byte) error
}
