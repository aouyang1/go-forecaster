package forecast

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelTablePrint(t *testing.T) {
	testData := map[string]struct {
		m        Model
		prefix   string
		indent   string
		expected string
	}{
		"no input": {
			m:      Model{},
			prefix: "",
			indent: "",
			expected: `Forecast:
Training End Time: 0001-01-01 00:00:00 +0000 UTC
Weights:
      Type Labels   Value
 Intercept        0.00000
`,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			err := td.m.TablePrint(&buf, td.prefix, td.indent)
			require.NoError(t, err)
			assert.Equal(t, td.expected, buf.String())
		})
	}
}
