package feature

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"
)

func TestSetSet(t *testing.T) {
	testData := map[string]struct {
		init     *Set
		f        Feature
		data     []float64
		expected *Set
	}{
		"initial set": {
			init: NewSet(),
			f:    NewEvent("blargh"),
			data: []float64{1, 2, 3, 4},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
		},
		"set with more data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f:    NewEvent("more"),
			data: []float64{1, 2, 3, 4, 5, 6},
			expected: &Set{
				m: 6,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4, 0, 0},
					"event_more":   {1, 2, 3, 4, 5, 6},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("more"),
				},
			},
		},
		"set with less data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f:    NewEvent("less"),
			data: []float64{1, 2},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
					"event_less":   {1, 2, 0, 0},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("less"),
				},
			},
		},
		"set with same data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f:    NewEvent("same"),
			data: []float64{5, 6, 7, 8},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
					"event_same":   {5, 6, 7, 8},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("same"),
				},
			},
		},
		"override": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f:    NewEvent("blargh"),
			data: []float64{5, 6, 7, 8},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {5, 6, 7, 8},
				},
				labels: []Feature{
					NewEvent("blargh"),
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			s := td.init.Set(td.f, td.data)
			assert.Equal(t, td.expected, s)
		})
	}
}

func TestSetDel(t *testing.T) {
	testData := map[string]struct {
		init     *Set
		f        Feature
		expected *Set
	}{
		"unknown feature": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f: NewEvent("asdf"),
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
		},
		"valid delete": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			f:        NewEvent("blargh"),
			expected: NewSet(),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			s := td.init.Del(td.f)
			assert.Equal(t, td.expected, s)
		})
	}
}

func TestSetUpdate(t *testing.T) {
	testData := map[string]struct {
		init     *Set
		next     *Set
		expected *Set
	}{
		"initial set": {
			init: NewSet(),
			next: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
		},
		"set with more data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			next: &Set{
				m: 6,
				set: map[string][]float64{
					"event_more": {1, 2, 3, 4, 5, 6},
				},
				labels: []Feature{NewEvent("more")},
			},
			expected: &Set{
				m: 6,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4, 0, 0},
					"event_more":   {1, 2, 3, 4, 5, 6},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("more"),
				},
			},
		},
		"set with less data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			next: &Set{
				m: 2,
				set: map[string][]float64{
					"event_less": {1, 2},
				},
				labels: []Feature{NewEvent("less")},
			},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
					"event_less":   {1, 2, 0, 0},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("less"),
				},
			},
		},
		"set with same data": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			next: &Set{
				m: 4,
				set: map[string][]float64{
					"event_same": {5, 6, 7, 8},
				},
				labels: []Feature{NewEvent("same")},
			},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
					"event_same":   {5, 6, 7, 8},
				},
				labels: []Feature{
					NewEvent("blargh"),
					NewEvent("same"),
				},
			},
		},
		"override": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			next: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {5, 6, 7, 8},
				},
				labels: []Feature{NewEvent("blargh")},
			},
			expected: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {5, 6, 7, 8},
				},
				labels: []Feature{
					NewEvent("blargh"),
				},
			},
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			s := td.init.Update(td.next)
			assert.Equal(t, td.expected, s)
		})
	}
}

func TestMatrix(t *testing.T) {
	testData := map[string]struct {
		init      *Set
		intercept bool
		expected  *mat.Dense
	}{
		"nil": {nil, true, nil},
		"initialized empty": {
			init:      &Set{},
			intercept: true,
			expected:  nil,
		},
		"with intercept": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{
					NewEvent("blargh"),
				},
			},
			intercept: true,
			expected: mat.NewDense(4, 2, []float64{
				1, 1,
				1, 2,
				1, 3,
				1, 4,
			}),
		},
		"without intercept": {
			init: &Set{
				m: 4,
				set: map[string][]float64{
					"event_blargh": {1, 2, 3, 4},
				},
				labels: []Feature{
					NewEvent("blargh"),
				},
			},
			intercept: false,
			expected:  mat.NewDense(4, 1, []float64{1, 2, 3, 4}),
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			res := td.init.Matrix(td.intercept)
			if td.expected == nil {
				assert.Nil(t, res)
				return
			}
			require.NotNil(t, res)
			resR, resC := res.Dims()
			expR, expC := td.expected.Dims()
			assert.Equal(t, expR, resR, "matrix rows")
			assert.Equal(t, expC, resC, "matrix columns")

			for i := 0; i < resR; i++ {
				assert.Equal(t, res.RawRowView(i), td.expected.RawRowView(i), fmt.Sprintf("row: %d", i))
			}
		})
	}
}

func TestRemoveZeroOnlyFeatures(t *testing.T) {
	s := NewSet().Set(
		NewTime("valid"),
		[]float64{1, 2, 3, 4},
	).Set(
		NewTime("only_zeros_1"),
		[]float64{0, 0, 0, 0},
	).Set(
		NewTime("only_zeros_2"),
		[]float64{0, 0, 0, 0},
	)

	vals, exists := s.Get(NewTime("valid"))
	assert.True(t, exists)
	assert.Equal(t, []float64{1, 2, 3, 4}, vals)

	vals, exists = s.Get(NewTime("only_zeros_1"))
	assert.True(t, exists)
	assert.Equal(t, []float64{0, 0, 0, 0}, vals)

	vals, exists = s.Get(NewTime("only_zeros_2"))
	assert.True(t, exists)
	assert.Equal(t, []float64{0, 0, 0, 0}, vals)

	s.RemoveZeroOnlyFeatures()

	vals, exists = s.Get(NewTime("valid"))
	assert.True(t, exists)
	assert.Equal(t, []float64{1, 2, 3, 4}, vals)

	vals, exists = s.Get(NewTime("only_zeros_1"))
	assert.False(t, exists)
	assert.Empty(t, vals)

	vals, exists = s.Get(NewTime("only_zeros_2"))
	assert.False(t, exists)
	assert.Empty(t, vals)
}
