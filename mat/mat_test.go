package mat

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestNewDenseFromArray(t *testing.T) {
	testData := map[string]struct {
		err error
		x   [][]float64
		m   int
		n   int
	}{
		"nil input": {
			mat.ErrZeroLength,
			nil,
			0, 0,
		},
		"empty input": {
			mat.ErrZeroLength,
			[][]float64{},
			0, 0,
		},
		"single element": {
			nil,
			[][]float64{{1}},
			1, 1,
		},
		"one row multiple cols": {
			nil,
			[][]float64{{1, 2, 3}},
			1, 3,
		},
		"multiple rows one col": {
			nil,
			[][]float64{{1}, {2}, {3}},
			3, 1,
		},
		"multiple rows and cols": {
			nil,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			2, 3,
		},
		"inconsistent cols": {
			ErrColMismatch,
			[][]float64{{1, 2, 3}, {4, 5}},
			0, 0,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer func() {
				r := recover()
				if td.err != nil && r != nil {
					err, ok := r.(error)
					require.True(t, ok, "panic is not an error")
					assert.ErrorAs(t, err, &td.err)
				}
			}()
			mx, err := NewDenseFromArray(td.x)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)

			m, n := mx.Dims()
			assert.Equal(t, td.m, m, "m")
			assert.Equal(t, td.n, n, "n")

			for ri, row := range td.x {
				assert.Equal(t, row, mat.Row(nil, ri, mx), "array")
			}
		})
	}
}
