package feature

import (
	"math"
	"strings"
	"time"

	"github.com/goccy/go-json"
	"gonum.org/v1/gonum/floats"
)

const (
	GrowthIntercept = "intercept"
	GrowthLinear    = "linear"
	GrowthQuadratic = "quadratic"
)

type Growth struct {
	Name string `json:"name"`

	str string `json:"-"`
}

func NewGrowth(name string) *Growth {
	strRep := "growth_" + name
	return &Growth{name, strRep}
}

// String returns the string representation of the changepoint feature
func (g Growth) String() string {
	return g.str
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
	g.str = "growth_" + g.Name
	return nil
}

func (g *Growth) Generate(epoch []float64, trainStartTime, trainEndTime time.Time) []float64 {
	// ensure that first feature is at 0
	epochTrainStart := float64(trainStartTime.UnixNano()) / 1e9

	// ensure last feature is at 1.0
	scale := float64(trainEndTime.Sub(trainStartTime).Nanoseconds()) / 1e9
	if scale == 0 {
		return nil
	}

	switch g.Name {
	case GrowthIntercept:
		ones := make([]float64, len(epoch))
		floats.AddConst(1.0, ones)
		return ones

	case GrowthLinear:
		linearGrowth := make([]float64, len(epoch))
		for i, epochT := range epoch {
			linearGrowth[i] = (float64(epochT) - epochTrainStart) / scale
		}
		return linearGrowth

	case GrowthQuadratic:
		quadraticGrowth := make([]float64, len(epoch))
		for i, epochT := range epoch {
			quadraticGrowth[i] = math.Pow((float64(epochT)-epochTrainStart)/scale, 2.0)
		}
		return quadraticGrowth

	}
	return nil
}

func Intercept() *Growth {
	return NewGrowth(GrowthIntercept)
}

func Linear() *Growth {
	return NewGrowth(GrowthLinear)
}

func Quadratic() *Growth {
	return NewGrowth(GrowthQuadratic)
}
