package feature

import (
	"strings"

	"github.com/goccy/go-json"
)

// Time feature representing a time based feature e.g. hour of day.
type Time struct {
	Name string `json:"name"`

	str string `json:"-"`
}

// NewTime creataes aa new time instance given a name
func NewTime(name string) *Time {
	strRep := "tfeat_" + name
	return &Time{name, strRep}
}

// String returns the string representationf of the time feature
func (t Time) String() string {
	return t.str
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
	var labelStr struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &labelStr); err != nil {
		return err
	}
	t.Name = labelStr.Name
	t.str = "tfeat_" + t.Name
	return nil
}
