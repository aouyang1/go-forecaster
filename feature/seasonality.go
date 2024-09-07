package feature

import (
	"fmt"
	"strconv"
	"strings"
)

type FourierComp string

const (
	FourierCompSin = "sin"
	FourierCompCos = "cos"
)

type Seasonality struct {
	Name        string      `json:"name"`
	FourierComp FourierComp `json:"fourier_component"`
	Order       int         `json:"order"`
}

func NewSeasonality(name string, fcomp FourierComp, order int) Seasonality {
	return Seasonality{name, fcomp, order}
}

func (s Seasonality) String() string {
	return fmt.Sprintf("seas_%s_%02d_%s", s.Name, s.Order, s.FourierComp)
}

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

func (s Seasonality) Type() FeatureType {
	return FeatureTypeSeasonality
}
