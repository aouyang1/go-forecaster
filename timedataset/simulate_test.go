package timedataset

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateT(t *testing.T) {
	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}

	numPnts := 7
	res := GenerateT(numPnts, 24*time.Hour, nowFunc)
	assert.Len(t, res, numPnts)

	assert.Equal(t, res[0], time.Date(1970, 1, 1, 0, 0, 0, 0, time.UTC))
	assert.Equal(t, res[numPnts-1], time.Date(1970, 1, 7, 0, 0, 0, 0, time.UTC))
}

func TestSeries(t *testing.T) {
	numPnts := 7
	s := Series(GenerateConstY(numPnts, 1))

	res := s.Add(GenerateConstY(numPnts, 2))
	require.Equal(t, Series([]float64{3, 3, 3, 3, 3, 3, 3}), res)

	nowFunc := func() time.Time {
		return time.Date(1970, 1, 8, 0, 0, 0, 0, time.UTC)
	}

	tSeries := GenerateT(numPnts, 24*time.Hour, nowFunc)
	s.SetConst(tSeries, 2.0,
		time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
		time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
	)
	assert.Equal(t, Series([]float64{3, 3, 2, 2, 3, 3, 3}), s)

	s.MaskWithWeekend(tSeries)
	assert.Equal(t, Series([]float64{0, 0, 2, 2, 0, 0, 0}), s)

	s.Add(GenerateConstY(numPnts, 1))
	s.MaskWithTimeRange(
		time.Date(1970, 1, 3, 0, 0, 0, 0, time.UTC),
		time.Date(1970, 1, 5, 0, 0, 0, 0, time.UTC),
		tSeries,
	)
	assert.Equal(t, Series([]float64{0, 0, 3, 3, 1, 0, 0}), s)
}
