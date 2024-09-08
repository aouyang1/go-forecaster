package feature

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Time struct {
	Name string `json:"name"`
}

func NewTime(name string) *Time {
	return &Time{name}
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

func (t Time) Type() FeatureType {
	return FeatureTypeTime
}

func (t Time) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = t.Name
	return res
}

func (t *Time) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, t)
}
