package timedataset

import (
	"math"
	"time"
)

type TimeSlice []time.Time

func (t TimeSlice) StartTime() time.Time {
	var startTime time.Time
	if len(t) < 1 {
		return startTime
	}
	return t[0]
}

func (t TimeSlice) EndTime() time.Time {
	var lastTime time.Time
	if len(t) < 1 {
		return lastTime
	}

	lastTime = t[len(t)-1]
	return lastTime
}

func (t TimeSlice) EstimateFreq() (time.Duration, error) {
	if len(t) < 2 {
		return 0, ErrCannotInferFreq
	}

	frequencies := make(map[time.Duration]int)
	for i := 1; i < len(t); i++ {
		delta := t[i].Sub(t[i-1])
		frequencies[delta] += 1
	}

	var maxCnt int
	maxDelta := time.Duration(math.MaxInt64)

	for delta, cnt := range frequencies {
		if cnt >= maxCnt && delta < maxDelta {
			maxCnt = cnt
			maxDelta = delta
		}
	}
	return maxDelta, nil
}
