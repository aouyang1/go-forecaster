package options

import (
	"log/slog"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
)

// MaxWeekendDurBuffer sets a limit of 1 day before or after the weekend begins at 00:00 Saturday
// or 00:00 Monday, respectively. Timezone is based on weekend option timezone override or dataset
// timezone
const MaxWeekendDurBuffer = 24 * time.Hour

// WeekendOptions lets us model weekends separately from weekdays.
type WeekendOptions struct {
	Enabled          bool          `json:"enabled"`
	TimezoneOverride string        `json:"timezone_override"`
	DurBefore        time.Duration `json:"duration_before"`
	DurAfter         time.Duration `json:"duration_after"`
}

func (w *WeekendOptions) Validate() {
	if w.DurBefore > MaxWeekendDurBuffer {
		w.DurBefore = MaxWeekendDurBuffer
	} else if w.DurBefore < -MaxWeekendDurBuffer {
		w.DurBefore = -MaxWeekendDurBuffer
	}

	if w.DurAfter > MaxWeekendDurBuffer {
		w.DurAfter = MaxWeekendDurBuffer
	} else if w.DurAfter < -MaxWeekendDurBuffer {
		w.DurAfter = -MaxWeekendDurBuffer
	}
}

func (w WeekendOptions) isWeekend(tPnt time.Time) bool {
	if w.DurBefore == 0 && w.DurAfter == 0 {
		wkday := tPnt.Weekday()
		return wkday == time.Saturday || wkday == time.Sunday
	}

	wkdayBefore := tPnt.Add(w.DurBefore).Weekday()
	wkdayAfter := tPnt.Add(-w.DurAfter).Weekday()

	wkdayBeforeValid := wkdayBefore == time.Saturday || wkdayBefore == time.Sunday
	wkdayAfterValid := wkdayAfter == time.Saturday || wkdayAfter == time.Sunday

	if w.DurBefore > 0 && w.DurAfter > 0 {
		return wkdayBeforeValid || wkdayAfterValid
	}

	return wkdayBeforeValid && wkdayAfterValid
}

func (w WeekendOptions) generateEventMask(t []time.Time, eFeat *feature.Set, winFunc func([]float64) []float64) {
	if !w.Enabled || len(t) < 2 {
		return
	}
	if w.TimezoneOverride != "" {
		locOverride, err := time.LoadLocation(w.TimezoneOverride)
		if err != nil {
			slog.Warn("invalid timezone location override for weekend options, using dataset timezone", "timezone_override", w.TimezoneOverride)
		} else {
			tShift := make([]time.Time, len(t))
			for i, val := range t {
				tShift[i] = val.In(locOverride)
			}
			t = tShift
		}
	}

	w.Validate()

	freq := t[1].Sub(t[0])
	start := t[0]
	end := t[len(t)-1]
	window := 2 * 24 * time.Hour

	// pad beginning
	numElem := int((window+w.DurBefore)/freq) + 1
	startIdx := numElem
	prefix := make([]time.Time, numElem)
	for i := 0; i < numElem; i++ {
		prefix[i] = start.Add(-time.Duration(numElem-i) * freq)
	}
	t = append(prefix, t...)

	// pad end
	numElem = int((window+w.DurAfter)/freq) + 1
	endIdx := len(t)
	suffix := make([]time.Time, numElem)
	for i := 0; i < numElem; i++ {
		suffix[i] = end.Add(time.Duration(i+1) * freq)
	}
	t = append(t, suffix...)

	weekendMask := generateEventMaskWithFunc(t, w.isWeekend, winFunc)

	// truncate result to start/end
	weekendMask = weekendMask[startIdx:endIdx]

	feat := feature.NewEvent(LabelEventWeekend)
	eFeat.Set(feat, weekendMask)
}
