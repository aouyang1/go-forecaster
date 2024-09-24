package feature

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
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
}

// NewSeasonality creates a new seasonality feature instance givenn a name, sin/cos component,
// and Fourier order
func NewSeasonality(name string, fcomp FourierComp, order int) *Seasonality {
	return &Seasonality{name, fcomp, order}
}

// String returns the string representation of the seasonality feature
func (s Seasonality) String() string {
	return fmt.Sprintf("seas_%s_%02d_%s", s.Name, s.Order, s.FourierComp)
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
	err := json.Unmarshal(data, &labelStr)
	if err != nil {
		return err
	}
	s.Name = labelStr.Name
	s.FourierComp = labelStr.FourierComp
	s.Order, err = strconv.Atoi(labelStr.Order)
	if err != nil {
		return err
	}
	return nil
}
