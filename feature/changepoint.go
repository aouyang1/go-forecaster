package feature

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ChangepointComp string

const (
	ChangepointCompBias  = "bias"
	ChangepointCompSlope = "slope"
)

type Changepoint struct {
	Name            string          `json:"name"`
	ChangepointComp ChangepointComp `json:"changepoint_component"`
}

func NewChangepoint(name string, comp ChangepointComp) *Changepoint {
	return &Changepoint{name, comp}
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

func (c Changepoint) Type() FeatureType {
	return FeatureTypeChangepoint
}

func (c Changepoint) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = c.Name
	res["changepoint_component"] = string(c.ChangepointComp)
	return res
}

func (c *Changepoint) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, c)
}
