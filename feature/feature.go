package feature

type FeatureType int

const (
	FeatureTypeChangepoint FeatureType = iota
	FeatureTypeSeasonality
	FeatureTypeTime
)

type Feature interface {
	String() string
	Get(string) (string, bool)
	Type() FeatureType
}
