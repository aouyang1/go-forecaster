package floatsunrolled

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func checkPanic(t *testing.T, err error) {
	r := recover()
	if r == nil {
		return
	}
	if err != nil {
		rErr, ok := r.(error)
		assert.True(t, ok)
		assert.EqualError(t, rErr, err.Error())
		return
	}

	assert.Nil(t, r)
}

func TestDot(t *testing.T) {
	testData := map[string]struct {
		a        []float64
		b        []float64
		err      error
		expected float64
	}{
		"length mismatch": {
			a:   []float64{1, 2, 3},
			b:   []float64{1, 2},
			err: ErrSliceLengthMismatch,
		},
		"length multiple invalid": {
			a:   []float64{1, 2, 3},
			b:   []float64{1, 2, 3},
			err: ErrSliceMul,
		},
		"valid": {
			a:        []float64{1, 2, 3, 4},
			b:        []float64{4, 3, 2, 1},
			expected: 20,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					if td.err != nil {
						err, ok := r.(error)
						assert.True(t, ok)
						assert.EqualError(t, err, td.err.Error())
						return
					}

					assert.Nil(t, r)
				}
			}()
			res := Dot(td.a, td.b)
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestAdd(t *testing.T) {
	testData := map[string]struct {
		dst      []float64
		s        []float64
		err      error
		expected []float64
	}{
		"length mismatch": {
			dst: []float64{1, 2, 3},
			s:   []float64{1, 2},
			err: ErrSliceLengthMismatch,
		},
		"length multiple invalid": {
			dst: []float64{1, 2, 3},
			s:   []float64{1, 2, 3},
			err: ErrSliceMul,
		},
		"valid": {
			dst:      []float64{1, 2, 3, 4},
			s:        []float64{4, 3, 2, 1},
			expected: []float64{5, 5, 5, 5},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer checkPanic(t, td.err)
			res := Add(td.dst, td.s)
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestSubTo(t *testing.T) {
	testData := map[string]struct {
		dst      []float64
		s        []float64
		t        []float64
		err      error
		expected []float64
	}{
		"length mismatch": {
			s:   []float64{1, 2, 3},
			t:   []float64{1, 2},
			err: ErrSliceLengthMismatch,
		},
		"length multiple invalid": {
			s:   []float64{1, 2, 3},
			t:   []float64{1, 2, 3},
			err: ErrSliceMul,
		},
		"valid no destination": {
			s:        []float64{1, 2, 3, 4},
			t:        []float64{4, 3, 2, 1},
			expected: []float64{-3, -1, 1, 3},
		},
		"valid with destination": {
			dst:      make([]float64, 4),
			s:        []float64{1, 2, 3, 4},
			t:        []float64{4, 3, 2, 1},
			expected: []float64{-3, -1, 1, 3},
		},
		"invalid destination": {
			dst: make([]float64, 3),
			s:   []float64{1, 2, 3, 4},
			t:   []float64{4, 3, 2, 1},
			err: ErrOutputSliceLengthMismatch,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer checkPanic(t, td.err)
			res := SubTo(td.dst, td.s, td.t)
			assert.Equal(t, td.expected, res)
		})
	}
}

func TestScaleTo(t *testing.T) {
	testData := map[string]struct {
		dst      []float64
		c        float64
		s        []float64
		err      error
		expected []float64
	}{
		"length multiple invalid": {
			s:   []float64{1, 2, 3},
			err: ErrSliceMul,
		},
		"valid no destination": {
			c:        3,
			s:        []float64{1, 2, 3, 4},
			expected: []float64{3, 6, 9, 12},
		},
		"valid with destination": {
			dst:      make([]float64, 4),
			c:        3,
			s:        []float64{1, 2, 3, 4},
			expected: []float64{3, 6, 9, 12},
		},
		"invalid destination": {
			dst: make([]float64, 3),
			c:   3,
			s:   []float64{1, 2, 3, 4},
			err: ErrOutputSliceLengthMismatch,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			defer checkPanic(t, td.err)
			res := ScaleTo(td.dst, td.c, td.s)
			assert.Equal(t, td.expected, res)
		})
	}
}
