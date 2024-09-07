package feature

import (
	"fmt"
	"strings"
)

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

func (t Time) Type() FeatureType {
	return FeatureTypeTime
}
