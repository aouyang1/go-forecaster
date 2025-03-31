package feature

import (
	"encoding/json"
	"fmt"
	"strings"
)

const (
	GrowthIntercept = "intercept"
	GrowthLinear    = "linear"
)

type Growth struct {
	Name string `json:"name"`
}

func NewGrowth(name string) *Growth {
	return &Growth{name}
}

// String returns the string representation of the changepoint feature
func (g Growth) String() string {
	return fmt.Sprintf("growth_%s", g.Name)
}

// Get returns the value of an arbitrary label annd returns the value along with whether
// the label exists
func (g Growth) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return g.Name, true
	}
	return "", false
}

// Type returns the type of this feature
func (g Growth) Type() FeatureType {
	return FeatureTypeGrowth
}

// Decode converts the feature into a map of label values
func (g Growth) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = g.Name
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a changepoint feature
func (g *Growth) UnmarshalJSON(data []byte) error {
	var labelStr struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &labelStr); err != nil {
		return err
	}
	g.Name = labelStr.Name
	return nil
}

func Intercept() *Growth {
	return NewGrowth(GrowthIntercept)
}
