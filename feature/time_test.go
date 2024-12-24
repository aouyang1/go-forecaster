package feature

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTimeString(t *testing.T) {
	feat := NewTime("blargh")
	expected := "tfeat_blargh"
	assert.Equal(t, expected, feat.String())
}

func TestTimeGet(t *testing.T) {
	feat := NewTime("blargh")

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

func TestTimeDecode(t *testing.T) {
	feat := NewTime("blargh")
	exp := map[string]string{
		"name": "blargh",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestTimeUnmarshalJSON(t *testing.T) {
	feat := NewTime("blargh")
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Time
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}
