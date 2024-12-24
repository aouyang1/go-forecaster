package feature

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSeasonalityString(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	expected := "seas_hod_02_cos"
	assert.Equal(t, expected, feat.String())
}

func TestSeasonalityGet(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)

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
			expVal:    "hod",
			expExists: true,
		},
		"exact match": {
			label:     "name",
			expVal:    "hod",
			expExists: true,
		},
		"fourier component": {
			label:     "fourier_component",
			expVal:    "cos",
			expExists: true,
		},
		"order": {
			label:     "order",
			expVal:    "2",
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

func TestSeasonalityDecode(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	exp := map[string]string{
		"name":              "hod",
		"fourier_component": "cos",
		"order":             "2",
	}
	assert.Equal(t, exp, feat.Decode())
}

func TestSeasonalityUnmarshalJSON(t *testing.T) {
	feat := NewSeasonality("hod", FourierCompCos, 2)
	out, err := json.Marshal(feat.Decode())
	require.NoError(t, err)

	var nextFeat Seasonality
	require.NoError(t, json.Unmarshal(out, &nextFeat))

	assert.Equal(t, feat, &nextFeat)
}
