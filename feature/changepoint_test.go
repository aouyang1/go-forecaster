package feature

import (
	"testing"

	"github.com/goccy/go-json"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChangepointString(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	expected := "chpnt_blargh_bias"
	assert.Equal(t, expected, feat.String())
}

func TestChangepointGet(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)

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
		"changepoint component": {
			label:     "changepoint_component",
			expVal:    "bias",
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

func TestChangepointDecode(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	exp := map[string]string{
		"name":                  "blargh",
		"changepoint_component": "bias",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestChangepointUnmarshalJSON(t *testing.T) {
	feat := NewChangepoint("blargh", ChangepointCompBias)
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Changepoint
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}
