package feature

type FeatureType int

const (
	FeatureTypeChangepoint FeatureType = iota
	FeatureTypeSeasonality
	FeatureTypeTime
)

func (f FeatureType) String() string {
	switch f {
	case FeatureTypeChangepoint:
		return "changepoint"
	case FeatureTypeSeasonality:
		return "seasonality"
	case FeatureTypeTime:
		return "time"
	default:
		return ""
	}
}

type Feature interface {
	String() string
	Get(string) (string, bool)
	Type() FeatureType
	Decode() map[string]string
	UnmarshalJSON([]byte) error
}
