package feature

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Event feature representing a point in time that we expect a jump or trend change in
// the training time series. The component is either of type bias (jump) or slope (trend).
type Event struct {
	Name string `json:"name"`
}

// NewEvent creates a new event instance given a name
func NewEvent(name string) *Event {
	return &Event{name}
}

// String returns the string representation of the event feature
func (e Event) String() string {
	return fmt.Sprintf("event_%s", e.Name)
}

// Get returns the value of an arbitrary label annd returns the value along with whether
// the label exists
func (e Event) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return e.Name, true
	}
	return "", false
}

// Type returns the type of this feature
func (e Event) Type() FeatureType {
	return FeatureTypeEvent
}

// Decode converts the feature into a map of label values
func (e Event) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = e.Name
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a event feature
func (e *Event) UnmarshalJSON(data []byte) error {
	var labelStr struct {
		Name string `json:"name"`
	}
	err := json.Unmarshal(data, &labelStr)
	if err != nil {
		return err
	}
	e.Name = labelStr.Name
	return nil
}
