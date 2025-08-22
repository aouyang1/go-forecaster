package options

import (
	"log/slog"
	"time"

	"github.com/aouyang1/go-forecaster/timedataset"
)

const (
	// MaxWeekendDurBuffer sets a limit of 1 day before or after the weekend begins at 00:00 Saturday
	// or 00:00 Monday, respectively. Timezone is based on weekend option timezone override or dataset
	// timezone
	MaxWeekendDurBuffer = 24 * time.Hour

	LabelEventWeekend = "weekend"
)

// WeekendOptions lets us model weekends separately from weekdays.
type WeekendOptions struct {
	Enabled          bool          `json:"enabled"`
	TimezoneOverride string        `json:"timezone_override"`
	DurBefore        time.Duration `json:"duration_before"`
	DurAfter         time.Duration `json:"duration_after"`
}

func (w *WeekendOptions) Name() string {
	return LabelEventWeekend
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

func (w *WeekendOptions) isWeekend(tPnt time.Time) bool {
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

func (w *WeekendOptions) GenerateMask(t []time.Time, windowName string) ([]float64, error) {
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

	startIdx := 0
	endIdx := len(t)
	// only perform padding on non rectangular windows as the weights are different at various points in time
	if windowName != "" && windowName != WindowRectangular {
		ts := timedataset.TimeSlice(t)
		freq, err := ts.EstimateFreq()
		if err != nil {
			return nil, err
		}

		start := ts.StartTime()
		end := ts.EndTime()
		window := 2 * 24 * time.Hour

		// pad beginning
		numElem := int((window+w.DurBefore)/freq) + 1
		startIdx = numElem
		prefix := make([]time.Time, numElem)
		for i := 0; i < numElem; i++ {
			prefix[i] = start.Add(-time.Duration(numElem-i) * freq)
		}
		t = append(prefix, t...)

		// pad end
		numElem = int((window+w.DurAfter)/freq) + 1
		endIdx = len(t)
		suffix := make([]time.Time, numElem)
		for i := 0; i < numElem; i++ {
			suffix[i] = end.Add(time.Duration(i+1) * freq)
		}
		t = append(t, suffix...)
	}

	winFunc := WindowFunc(windowName)
	weekendMask := generateEventMaskWithFunc(t, w.isWeekend, winFunc)

	// truncate result to start/end
	weekendMask = weekendMask[startIdx:endIdx]
	return weekendMask, nil
}
