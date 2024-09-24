package feature

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Time feature representing a time based feature e.g. hour of day.
type Time struct {
	Name string `json:"name"`
}

// NewTime creataes aa new time instance given a name
func NewTime(name string) *Time {
	return &Time{name}
}

// String returns the string representationf of the time feature
func (t Time) String() string {
	return fmt.Sprintf("tfeat_%s", t.Name)
}

// Get returns the value of an arbitrary label annd returns the value along with whether
// the label exists
func (t Time) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return t.Name, true
	}
	return "", false
}

// Type returns the type of this feature
func (t Time) Type() FeatureType {
	return FeatureTypeTime
}

// Decode converts the feature into a map of label values
func (t Time) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = t.Name
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a time feature
func (t *Time) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, t)
}
