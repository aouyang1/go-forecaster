package feature

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

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
