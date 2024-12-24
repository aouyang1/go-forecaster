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

// Changepoint feature representing a point in time that we expect a jump or trend change in
// the training time series. The component is either of type bias (jump) or slope (trend).
type Changepoint struct {
	Name            string          `json:"name"`
	ChangepointComp ChangepointComp `json:"changepoint_component"`
}

// NewChangepoint creates a new changepoint instance given a name and changepoint component
// type
func NewChangepoint(name string, comp ChangepointComp) *Changepoint {
	return &Changepoint{name, comp}
}

// String returns the string representation of the changepoint feature
func (c Changepoint) String() string {
	return fmt.Sprintf("chpnt_%s_%s", c.Name, c.ChangepointComp)
}

// Get returns the value of an arbitrary label annd returns the value along with whether
// the label exists
func (c Changepoint) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return c.Name, true
	case "changepoint_component":
		return string(c.ChangepointComp), true
	}
	return "", false
}

// Type returns the type of this feature
func (c Changepoint) Type() FeatureType {
	return FeatureTypeChangepoint
}

// Decode converts the feature into a map of label values
func (c Changepoint) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = c.Name
	res["changepoint_component"] = string(c.ChangepointComp)
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a changepoint feature
func (c *Changepoint) UnmarshalJSON(data []byte) error {
	var labelStr struct {
		Name            string          `json:"name"`
		ChangepointComp ChangepointComp `json:"changepoint_component"`
	}
	if err := json.Unmarshal(data, &labelStr); err != nil {
		return err
	}
	c.Name = labelStr.Name
	c.ChangepointComp = labelStr.ChangepointComp
	return nil
}
