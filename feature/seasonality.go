package feature

import (
	"math"
	"strconv"
	"strings"

	"github.com/goccy/go-json"
)

type FourierComp string

const (
	FourierCompSin = "sin"
	FourierCompCos = "cos"
)

// Seasonality feature representing a single seasonality component. A seasonality
// component consists of a name, component (sin or cos), and order (fourier order).
type Seasonality struct {
	Name        string      `json:"name"`
	FourierComp FourierComp `json:"fourier_component"`
	Order       int         `json:"order"`

	str string `json:"-"`
}

// NewSeasonality creates a new seasonality feature instance givenn a name, sin/cos component,
// and Fourier order
func NewSeasonality(name string, fcomp FourierComp, order int) *Seasonality {
	strRep := "seas_" + name + "_" + strconv.Itoa(order) + "_" + string(fcomp)
	return &Seasonality{name, fcomp, order, strRep}
}

// String returns the string representation of the seasonality feature
func (s Seasonality) String() string {
	return s.str
}

// Get returns the value of an arbitrary label and returns the value along with whether
// the label exists
func (s Seasonality) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return s.Name, true
	case "fourier_component":
		return string(s.FourierComp), true
	case "order":
		return strconv.Itoa(s.Order), true
	}
	return "", false
}

// Type returns the type of this feature
func (s Seasonality) Type() FeatureType {
	return FeatureTypeSeasonality
}

// Decode converts the feature innto a map of label values
func (s Seasonality) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = s.Name
	res["fourier_component"] = string(s.FourierComp)
	res["order"] = strconv.Itoa(s.Order)
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a seasonality feature
func (s *Seasonality) UnmarshalJSON(data []byte) error {
	var labelStr struct {
		Name        string      `json:"name"`
		FourierComp FourierComp `json:"fourier_component"`
		Order       string      `json:"order"`
	}
	if err := json.Unmarshal(data, &labelStr); err != nil {
		return err
	}
	order, err := strconv.Atoi(labelStr.Order)
	if err != nil {
		return err
	}
	s.Name = labelStr.Name
	s.FourierComp = labelStr.FourierComp
	s.Order = order
	s.str = "seas_" + s.Name + "_" + strconv.Itoa(s.Order) + "_" + string(s.FourierComp)
	return nil
}

func (s *Seasonality) Generate(t []float64, order int, period float64) []float64 {
	omega := 2.0 * math.Pi * float64(order) / period
	feat := make([]float64, len(t))
	for i, tFeat := range t {
		rad := omega * tFeat
		switch s.FourierComp {
		case FourierCompSin:
			feat[i] = math.Sin(rad)
		case FourierCompCos:
			feat[i] = math.Cos(rad)
		}
	}
	return feat
}
