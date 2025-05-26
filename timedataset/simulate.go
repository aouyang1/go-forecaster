package timedataset

import (
	"math"
	"math/rand/v2"
	"time"

	"gonum.org/v1/gonum/floats"
)

func GenerateT(n int, interval time.Duration, nowFunc func() time.Time) []time.Time {
	t := make([]time.Time, 0, n)
	ct := time.Unix(nowFunc().Unix()/60*60, 0).Add(-time.Duration(n) * interval).UTC()
	for i := 0; i < n; i++ {
		t = append(t, ct.Add(interval*time.Duration(i)))
	}
	return t
}

type Series []float64

func (s Series) Add(src Series) Series {
	floats.Add(s, src)
	return s
}

func (s Series) SetConst(t []time.Time, val float64, start, end time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		if (t[i].After(start) || t[i].Equal(start)) && t[i].Before(end) {
			s[i] = val
		}
	}
	return s
}

func (s Series) MaskWithWeekend(t []time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		switch t[i].Weekday() {
		case time.Saturday, time.Sunday:
			continue
		default:
			s[i] = 0.0
		}
	}
	return s
}

func (s Series) MaskWithTimeRange(start, end time.Time, t []time.Time) Series {
	n := len(s)
	for i := 0; i < n; i++ {
		if t[i].Before(start) || t[i].After(end) {
			s[i] = 0.0
		}
	}
	return s
}

func GenerateConstY(n int, val float64) Series {
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		y = append(y, val)
	}
	return Series(y)
}

func GenerateWaveY(t []time.Time, amp, periodSec, order, timeOffset float64) Series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		val := amp * math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset))
		y = append(y, val)
	}
	return Series(y)
}

func GenerateNoise(t []time.Time, noiseScale, amp, periodSec, order, timeOffset float64) Series {
	n := len(t)
	y := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		scale := (noiseScale + amp*math.Sin(2.0*math.Pi*order/periodSec*(float64(t[i].Unix())+timeOffset)))
		y = append(y, rand.NormFloat64()*scale)
	}
	return Series(y)
}

func GenerateChange(t []time.Time, chpt time.Time, bias, slope float64) Series {
	n := len(t)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		if t[i].After(chpt) || t[i].Equal(chpt) {
			jump := bias + slope*t[i].Sub(chpt).Minutes()
			y[i] = jump
		}
	}
	return Series(y)
}

func GeneratePulseY(t []time.Time, amp, periodSec, order, timeOffset, duty float64) Series {
	n := len(t)
	y := make([]float64, 0, n)
	cycleCutoff := 1.0 - duty/2.0
	for i := 0; i < n; i++ {
		cyclePos := math.Cos(2.0 * math.Pi * order / periodSec * (float64(t[i].Unix()) + timeOffset))
		val := 0.0
		if cyclePos >= cycleCutoff {
			val = amp
		}

		y = append(y, val)
	}
	return Series(y)
}
