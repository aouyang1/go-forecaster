package feature

import (
	"fmt"
	"strconv"
	"strings"
)

type Feature interface {
	String() string
	Get(string) (string, bool)
}

type Time struct {
	Name string `json:"name"`
}

func NewTime(name string) Time {
	return Time{name}
}

func (t Time) String() string {
	return fmt.Sprintf("tfeat_%s", t.Name)
}

func (t Time) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return t.Name, true
	}
	return "", false
}

type ChangepointComp string

const (
	ChangepointCompBias  = "bias"
	ChangepointCompSlope = "slope"
)

type Changepoint struct {
	Name            string          `json:"name"`
	ChangepointComp ChangepointComp `json:"changepoint_component"`
}

func NewChangepoint(name string, comp ChangepointComp) Changepoint {
	return Changepoint{name, comp}
}

func (c Changepoint) String() string {
	return fmt.Sprintf("chpnt_%s_%s", c.Name, c.ChangepointComp)
}

func (c Changepoint) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return c.Name, true
	case "changepoint_component":
		return string(c.ChangepointComp), true
	}
	return "", false
}

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
