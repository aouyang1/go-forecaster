package array

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew1D(t *testing.T) {
	testData := map[string]struct {
		x   []float64
		arr []float64
		m   int
	}{
		"nil input": {
			nil,
			[]float64{},
			0,
		},
		"empty input": {
			[]float64{},
			[]float64{},
			0,
		},
		"single element": {
			[]float64{1},
			[]float64{1},
			1,
		},
		"multiple elements": {
			[]float64{1, 2, 3},
			[]float64{1, 2, 3},
			3,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr := New1D(td.x)
			assert.Equal(t, td.arr, arr.arr, "array")
			assert.Equal(t, td.m, arr.m, "m")
			assert.Equal(t, 1, arr.n, "n")
		})
	}
}

func TestNew2D(t *testing.T) {
	testData := map[string]struct {
		x   [][]float64
		err error
		arr []float64
		m   int
		n   int
	}{
		"nil input": {
			nil,
			nil,
			[]float64{},
			0, 0,
		},
		"empty input": {
			[][]float64{},
			nil,
			[]float64{},
			0, 0,
		},
		"single element": {
			[][]float64{{1}},
			nil,
			[]float64{1},
			1, 1,
		},
		"one row multiple cols": {
			[][]float64{{1, 2, 3}},
			nil,
			[]float64{1, 2, 3},
			1, 3,
		},
		"multiple rows one col": {
			[][]float64{{1}, {2}, {3}},
			nil,
			[]float64{1, 2, 3},
			3, 1,
		},
		"multiple rows and cols": {
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			nil,
			[]float64{1, 4, 2, 5, 3, 6},
			2, 3,
		},
		"inconsistent cols": {
			[][]float64{{1, 2, 3}, {4, 5}},
			ErrColMismatch,
			nil,
			0, 0,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)
			assert.Equal(t, td.arr, arr.arr, "array")
			assert.Equal(t, td.m, arr.m, "m")
			assert.Equal(t, td.n, arr.n, "n")
		})
	}
}

func TestGet(t *testing.T) {
	testData := map[string]struct {
		err      error
		x        [][]float64
		r, c     int
		expected float64
	}{
		"single element": {
			nil,
			[][]float64{{1}},
			0, 0,
			1.0,
		},
		"multiple rows and cols": {
			nil,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			1, 2,
			6.0,
		},
		"row out of bounds": {
			ErrRowOutOfBounds,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			2, 1,
			0.0,
		},
		"col out of bounds": {
			ErrColOutOfBounds,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			1, 3,
			0.0,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)

			val, err := arr.Get(td.r, td.c)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)

			assert.Equal(t, td.expected, val)
		})
	}
}

func TestGetCol(t *testing.T) {
	testData := map[string]struct {
		err      error
		x        [][]float64
		expected []float64
		c        int
	}{
		"single element": {
			nil,
			[][]float64{{1}},
			[]float64{1.0},
			0,
		},
		"multiple rows and cols": {
			nil,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[]float64{2, 5},
			1,
		},
		"col out of bounds": {
			ErrColOutOfBounds,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			nil,
			3,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)

			values, err := arr.GetCol(td.c)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)

			assert.Equal(t, td.expected, values)
		})
	}
}

func TestGetRow(t *testing.T) {
	testData := map[string]struct {
		err      error
		x        [][]float64
		expected []float64
		r        int
	}{
		"single element": {
			nil,
			[][]float64{{1}},
			[]float64{1.0},
			0,
		},
		"multiple rows and cols": {
			nil,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[]float64{4, 5, 6},
			1,
		},
		"row out of bounds": {
			ErrRowOutOfBounds,
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			nil,
			2,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)

			values, err := arr.GetRow(td.r)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)

			assert.Equal(t, td.expected, values)
		})
	}
}

func TestT(t *testing.T) {
	testData := map[string]struct {
		x   [][]float64
		arr [][]float64
		m   int
		n   int
	}{
		"empty input": {
			[][]float64{},
			[][]float64{},
			0, 0,
		},
		"single element": {
			[][]float64{{1}},
			[][]float64{{1}},
			1, 1,
		},
		"one row multiple cols": {
			[][]float64{{1, 2, 3}},
			[][]float64{{1}, {2}, {3}},
			3, 1,
		},
		"multiple rows one col": {
			[][]float64{{1}, {2}, {3}},
			[][]float64{{1, 2, 3}},
			1, 3,
		},
		"multiple rows and cols n>m": {
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 4}, {2, 5}, {3, 6}},
			3, 2,
		},
		"multiple rows and cols m>n": {
			[][]float64{{1, 4}, {2, 5}, {3, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			2, 3,
		},
		"multiple rows and cols m==n": {
			[][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			[][]float64{{1, 4, 7}, {2, 5, 8}, {3, 6, 9}},
			3, 3,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)

			res := arr.T()
			assert.Equal(t, td.arr, res.ToSlice(), "array")
			assert.Equal(t, td.m, res.m, "m")
			assert.Equal(t, td.n, res.n, "n")
		})
	}
}

func TestFlatten(t *testing.T) {
	testData := map[string]struct {
		x        [][]float64
		expected []float64
	}{
		"nil input": {
			nil,
			[]float64{},
		},
		"empty input": {
			[][]float64{},
			[]float64{},
		},
		"single element": {
			[][]float64{{1}},
			[]float64{1},
		},
		"one row multiple cols": {
			[][]float64{{1, 2, 3}},
			[]float64{1, 2, 3},
		},
		"multiple rows one col": {
			[][]float64{{1}, {2}, {3}},
			[]float64{1, 2, 3},
		},
		"multiple rows and cols": {
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[]float64{1, 2, 3, 4, 5, 6},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)
			assert.Equal(t, td.expected, arr.Flatten())
		})
	}
}

func TestToSlice(t *testing.T) {
	testData := map[string]struct {
		x        [][]float64
		expected [][]float64
	}{
		"nil input": {
			nil,
			[][]float64{},
		},
		"empty input": {
			[][]float64{},
			[][]float64{},
		},
		"single element": {
			[][]float64{{1}},
			[][]float64{{1}},
		},
		"one row multiple cols": {
			[][]float64{{1, 2, 3}},
			[][]float64{{1, 2, 3}},
		},
		"multiple rows one col": {
			[][]float64{{1}, {2}, {3}},
			[][]float64{{1}, {2}, {3}},
		},
		"multiple rows and cols": {
			[][]float64{{1, 2, 3}, {4, 5, 6}},
			[][]float64{{1, 2, 3}, {4, 5, 6}},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := New2D(td.x)
			require.Nil(t, err)
			assert.Equal(t, td.expected, arr.ToSlice())
		})
	}
}

func TestExtend(t *testing.T) {
	testData := map[string]struct {
		a        *Array
		b        *Array
		err      error
		expected *Array
	}{
		"nil a": {
			nil,
			&Array{
				arr: []float64{3, 3, 3},
				m:   3, n: 1,
			},
			ErrUninitializedArray,
			nil,
		},
		"nil b": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			nil,
			ErrUninitializedArray,
			nil,
		},
		"extend by 1 col": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 3, 3},
				m:   3, n: 1,
			},
			nil,
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1, 3, 3, 3},
				m:   3, n: 3,
			},
		},
		"extend by 2 cols": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 3, 3, 4, 4, 4},
				m:   3, n: 2,
			},
			nil,
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4},
				m:   3, n: 4,
			},
		},
		"incompatible extend": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 3, 4, 4},
				m:   2, n: 2,
			},
			ErrRowMismatch,
			nil,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := Extend(td.a, td.b)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)
			assert.Equal(t, td.expected, arr)
		})
	}
}

func TestAppend(t *testing.T) {
	testData := map[string]struct {
		a        *Array
		b        *Array
		err      error
		expected *Array
	}{
		"nil a": {
			nil,
			&Array{
				arr: []float64{3, 3, 3},
				m:   3, n: 1,
			},
			ErrUninitializedArray,
			nil,
		},
		"nil b": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			nil,
			ErrUninitializedArray,
			nil,
		},
		"extend by 1 row": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 3},
				m:   1, n: 2,
			},
			nil,
			&Array{
				arr: []float64{0, 0, 0, 3, 1, 1, 1, 3},
				m:   4, n: 2,
			},
		},
		"extend by 2 rows": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 4, 3, 4},
				m:   2, n: 2,
			},
			nil,
			&Array{
				arr: []float64{0, 0, 0, 3, 4, 1, 1, 1, 3, 4},
				m:   5, n: 2,
			},
		},
		"incompatible extend": {
			&Array{
				arr: []float64{0, 0, 0, 1, 1, 1},
				m:   3, n: 2,
			},
			&Array{
				arr: []float64{3, 4, 5, 3, 4, 5},
				m:   3, n: 3,
			},
			ErrColMismatch,
			nil,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			arr, err := Append(td.a, td.b)
			if td.err != nil {
				require.ErrorAs(t, err, &td.err)
				return
			}
			require.Nil(t, err)
			assert.Equal(t, td.expected, arr)
		})
	}
}
