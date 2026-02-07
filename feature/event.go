package feature

import (
	"log/slog"
	"strings"
	"time"

	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/goccy/go-json"
	"gonum.org/v1/gonum/dsp/window"
)

// Event feature representing a point in time that we expect a jump or trend change in
// the training time series. The component is either of type bias (jump) or slope (trend).
type Event struct {
	Name string `json:"name"`

	str string `json:"-"`
}

// NewEvent creates a new event instance given a name
func NewEvent(name string) *Event {
	strRep := "event_" + name
	return &Event{name, strRep}
}

// String returns the string representation of the event feature
func (e Event) String() string {
	return e.str
}

// Get returns the value of an arbitrary label annd returns the value along with whether
// the label exists
func (e Event) Get(label string) (string, bool) {
	switch strings.ToLower(label) {
	case "name":
		return e.Name, true
	}
	return "", false
}

// Type returns the type of this feature
func (e Event) Type() FeatureType {
	return FeatureTypeEvent
}

// Decode converts the feature into a map of label values
func (e Event) Decode() map[string]string {
	res := make(map[string]string)
	res["name"] = e.Name
	return res
}

// UnmarshalJSON is the custom unmarshalling to convert a map[string]string
// to a event feature
func (e *Event) UnmarshalJSON(data []byte) error {
	var labelStr struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &labelStr); err != nil {
		return err
	}
	e.Name = labelStr.Name
	e.str = "event_" + e.Name
	return nil
}

func (e *Event) Generate(t []time.Time, window string, evGenerator EventGenerator) ([]float64, error) {
	return evGenerator.Generate(t, window)
}

type EventGenerator interface {
	Generate(t []time.Time, window string) ([]float64, error)
}

type EventStandard struct {
	Start time.Time
	End   time.Time
}

func (e EventStandard) Generate(t []time.Time, window string) ([]float64, error) {
	ts := timedataset.TimeSlice(t)
	freq, err := ts.EstimateFreq()
	if err != nil {
		return nil, err
	}
	start := ts.StartTime()
	end := ts.EndTime()

	startIdx := 0
	endIdx := len(t)

	winFunc := WindowFunc(window)

	if window != "" && window != WindowRectangular {
		t, startIdx, endIdx = e.padTime(t, start, end, freq)
	}

	eventMask := generateEventMaskWithFunc(t, func(tPnt time.Time) bool {
		return (tPnt.After(e.Start) || tPnt.Equal(e.Start)) && tPnt.Before(e.End)
	}, winFunc)

	// truncate result to start/end
	eventMask = eventMask[startIdx:endIdx]
	return eventMask, nil
}

// padTime padding is done for the sake of applying a relevant window to the mask. this
// is only necessary is using anything beyond a rect window function
func (e EventStandard) padTime(t []time.Time, start, end time.Time, freq time.Duration) ([]time.Time, int, int) {
	// pad beginning
	var startIdx int
	if e.Start.Before(start) {
		diff := start.Sub(e.Start)
		numElem := int(diff/freq) + 1
		startIdx = numElem

		prefix := make([]time.Time, numElem)
		for i := range numElem {
			prefix[i] = start.Add(-time.Duration(numElem-i) * freq)
		}
		t = append(prefix, t...)
	}

	// pad end
	endIdx := len(t)
	if e.End.After(end) {
		diff := e.End.Sub(end)
		numElem := int(diff/freq) + 1

		suffix := make([]time.Time, numElem)
		for i := range numElem {
			suffix[i] = end.Add(time.Duration(i+1) * freq)
		}
		t = append(t, suffix...)
	}
	return t, startIdx, endIdx
}

const (
	// MaxWeekendDurBuffer sets a limit of 1 day before or after the weekend begins at 00:00 Saturday
	// or 00:00 Monday, respectively. Timezone is based on weekend option timezone override or dataset
	// timezone
	MaxWeekendDurBuffer = 24 * time.Hour

	EventNameWeekend = "weekend"
)

type EventWeekend struct {
	TimezoneOverride string
	DurBefore        time.Duration
	DurAfter         time.Duration
}

func (e *EventWeekend) Validate() {
	if e.DurBefore > MaxWeekendDurBuffer {
		e.DurBefore = MaxWeekendDurBuffer
	} else if e.DurBefore < -MaxWeekendDurBuffer {
		e.DurBefore = -MaxWeekendDurBuffer
	}

	if e.DurAfter > MaxWeekendDurBuffer {
		e.DurAfter = MaxWeekendDurBuffer
	} else if e.DurAfter < -MaxWeekendDurBuffer {
		e.DurAfter = -MaxWeekendDurBuffer
	}
}

func (e EventWeekend) isWeekend(tPnt time.Time) bool {
	if e.DurBefore == 0 && e.DurAfter == 0 {
		wkday := tPnt.Weekday()
		return wkday == time.Saturday || wkday == time.Sunday
	}

	wkdayBefore := tPnt.Add(e.DurBefore).Weekday()
	wkdayAfter := tPnt.Add(-e.DurAfter).Weekday()

	wkdayBeforeValid := wkdayBefore == time.Saturday || wkdayBefore == time.Sunday
	wkdayAfterValid := wkdayAfter == time.Saturday || wkdayAfter == time.Sunday

	if e.DurBefore > 0 && e.DurAfter > 0 {
		return wkdayBeforeValid || wkdayAfterValid
	}

	return wkdayBeforeValid && wkdayAfterValid
}

func (e EventWeekend) Generate(t []time.Time, windowName string) ([]float64, error) {
	if e.TimezoneOverride != "" {
		locOverride, err := time.LoadLocation(e.TimezoneOverride)
		if err != nil {
			slog.Warn("invalid timezone location override for weekend options, using dataset timezone", "timezone_override", e.TimezoneOverride)
		} else {
			tShift := make([]time.Time, len(t))
			for i, val := range t {
				tShift[i] = val.In(locOverride)
			}
			t = tShift
		}
	}

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
		numElem := int((window+e.DurBefore)/freq) + 1
		startIdx = numElem
		prefix := make([]time.Time, numElem)
		for i := 0; i < numElem; i++ {
			prefix[i] = start.Add(-time.Duration(numElem-i) * freq)
		}
		t = append(prefix, t...)

		// pad end
		numElem = int((window+e.DurAfter)/freq) + 1
		endIdx = len(t)
		suffix := make([]time.Time, numElem)
		for i := 0; i < numElem; i++ {
			suffix[i] = end.Add(time.Duration(i+1) * freq)
		}
		t = append(t, suffix...)
	}

	winFunc := WindowFunc(windowName)
	weekendMask := generateEventMaskWithFunc(t, e.isWeekend, winFunc)

	// truncate result to start/end
	weekendMask = weekendMask[startIdx:endIdx]
	return weekendMask, nil
}

func generateEventMaskWithFunc(t []time.Time, maskCond func(tPnt time.Time) bool, windowFunc func(seq []float64) []float64) []float64 {
	mask := make([]float64, len(t))
	var maskSpans [][2]int
	var inMask bool
	var maskSpan [2]int
	for i, tPnt := range t {
		if maskCond(tPnt) {
			if !inMask {
				inMask = true
				maskSpan[0] = i
			}
			mask[i] = 1.0
			continue
		}
		if inMask {
			inMask = false
			maskSpan[1] = i
			maskSpans = append(maskSpans, maskSpan)
		}
	}
	if inMask {
		maskSpan[1] = len(t)
		maskSpans = append(maskSpans, maskSpan)
	}

	for _, maskSpan := range maskSpans {
		windowFunc(mask[maskSpan[0]:maskSpan[1]])
	}
	return mask
}

const (
	WindowBartlettHann    = "bartlett_hann"
	WindowBlackman        = "blackman"
	WindowBlackmanHarris  = "blackman_harris"
	WindowBlackmanNuttall = "blackman_nuttall"
	WindowFlatTop         = "flat_top"
	WindowHamming         = "hamming"
	WindowHann            = "hann"
	WindowLanczos         = "lanczos"
	WindowNuttall         = "nuttall"
	WindowRectangular     = "rectangular"
	WindowSine            = "sine"
	WindowTriangular      = "triangular"
	WindowTukey           = "tukey"
)

var WindowParamTukeyAlpha = 0.95

func WindowFunc(name string) func(seq []float64) []float64 {
	var winFunc func(seq []float64) []float64
	switch name {
	case WindowBartlettHann:
		winFunc = window.BartlettHann
	case WindowBlackman:
		winFunc = window.Blackman
	case WindowBlackmanHarris:
		winFunc = window.BlackmanHarris
	case WindowBlackmanNuttall:
		winFunc = window.BlackmanNuttall
	case WindowFlatTop:
		winFunc = window.FlatTop
	case WindowHamming:
		winFunc = window.Hamming
	case WindowHann:
		winFunc = window.Hann
	case WindowLanczos:
		winFunc = window.Lanczos
	case WindowNuttall:
		winFunc = window.Nuttall
	case WindowRectangular:
		winFunc = window.Rectangular
	case WindowSine:
		winFunc = window.Sine
	case WindowTriangular:
		winFunc = window.Triangular
	case WindowTukey:
		winFunc = func(seq []float64) []float64 {
			return window.Tukey{Alpha: WindowParamTukeyAlpha}.Transform(seq)
		}
	default:
		winFunc = window.Rectangular
	}
	return winFunc
}
