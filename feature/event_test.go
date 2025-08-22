package feature

import (
	"testing"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEventString(t *testing.T) {
	feat := NewEvent("blargh")
	expected := "event_blargh"
	assert.Equal(t, expected, feat.String())
}

func TestEventGet(t *testing.T) {
	feat := NewEvent("blargh")

	testData := map[string]struct {
		label     string
		expVal    string
		expExists bool
	}{
		"unknown": {
			label: "unknown",
		},
		"capitalized": {
			label:     "NAME",
			expVal:    "blargh",
			expExists: true,
		},
		"exact match": {
			label:     "name",
			expVal:    "blargh",
			expExists: true,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			val, exists := feat.Get(td.label)
			assert.Equal(t, td.expExists, exists, "exists")
			assert.Equal(t, td.expVal, val, "value")
		})
	}
}

func TestEventDecode(t *testing.T) {
	feat := NewEvent("blargh")
	exp := map[string]string{
		"name": "blargh",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestEventUnmarshalJSON(t *testing.T) {
	feat := NewEvent("blargh")
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Event
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}
