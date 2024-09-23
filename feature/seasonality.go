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

type Seasonality struct {
	Name        string      `json:"name"`
	FourierComp FourierComp `json:"fourier_component"`
	Order       int         `json:"order"`
}

func NewSeasonality(name string, fcomp FourierComp, order int) *Seasonality {
	return &Seasonality{name, fcomp, order}
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

func (s Seasonality) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = s.Name
	res["fourier_component"] = string(s.FourierComp)
	res["order"] = strconv.Itoa(s.Order)
	return res
}

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
