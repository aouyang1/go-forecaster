package options

import (
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/util"
	"github.com/aouyang1/go-forecaster/timedataset"
	"github.com/rickar/cal/v2"
	"github.com/rickar/cal/v2/us"
)

var (
	ErrStartAfterEnd = errors.New("event start time is after end time")
	ErrUnsetTime     = errors.New("unset event start or end time")
	ErrNoEventName   = errors.New("no event name")
)

// Event represents a time span to model separately for bias and for seasonality
// changes.
type Event struct {
	Name  string
	Start time.Time
	End   time.Time
}

func NewEvent(name string, start, end time.Time) Event {
	return Event{
		Name:  name,
		Start: start,
		End:   end,
	}
}

func (e *Event) Valid() error {
	if e.Start.IsZero() || e.End.IsZero() {
		return ErrUnsetTime
	}
	if e.Start.After(e.End) {
		return ErrStartAfterEnd
	}
	if e.Name == "" {
		return ErrNoEventName
	}
	return nil
}

// padTime padding is done for the sake of applying a relevant window to the mask. this
// is only necessary is using anything beyond a rect window function
func (e *Event) padTime(t []time.Time, start, end time.Time, freq time.Duration) ([]time.Time, int, int) {
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

func Christmas(start, end time.Time, durBefore, durAfter time.Duration) []Event {
	return Holiday(us.ChristmasDay, start, end, durBefore, durAfter)
}

func Thanksgiving(start, end time.Time, durBefore, durAfter time.Duration) []Event {
	return Holiday(us.ThanksgivingDay, start, end, durBefore, durAfter)
}

func Holiday(hol *cal.Holiday, start, end time.Time, durBefore, durAfter time.Duration) []Event {
	startLoc := start.Location()

	events := []Event{}
	for i := start.Year(); i <= end.Year(); i++ {
		_, observed := hol.Calc(i)
		_, offset := observed.Zone()
		_, startOffset := start.Zone()

		observed = observed.Add(time.Duration(offset) * time.Second).In(startLoc).Add(time.Duration(-startOffset) * time.Second)

		if (observed.After(start) || observed.Equal(start)) && (observed.Before(end) || observed.Equal(end)) {
			events = append(events, Event{
				Name:  strings.ReplaceAll(hol.Name, " ", "_") + "_" + strconv.Itoa(i),
				Start: observed.Add(-durBefore),
				End:   observed.Add(24 * time.Hour).Add(durAfter),
			})
		}
	}
	return events
}

type EventOptions struct {
	Events []Event `json:"events"`
}

func (e EventOptions) generateEventMask(t []time.Time, eFeat *feature.Set, window string) {
	if len(t) < 2 {
		return
	}

	winFunc := WindowFunc(window)

	ts := timedataset.TimeSlice(t)
	freq, err := ts.EstimateFreq()
	if err != nil {
		panic(err)
	}
	start := ts.StartTime()
	end := ts.EndTime()
	for _, ev := range e.Events {
		if err := ev.Valid(); err != nil {
			slog.Warn("not separately modelling invalid event", "name", ev.Name, "error", err.Error())
			continue
		}

		evT := t
		startIdx := 0
		endIdx := len(t)
		if window != "" && window != WindowRectangular {
			evT, startIdx, endIdx = ev.padTime(t, start, end, freq)
		}

		eventMask := generateEventMaskWithFunc(evT, func(tPnt time.Time) bool {
			return (tPnt.After(ev.Start) || tPnt.Equal(ev.Start)) && tPnt.Before(ev.End)
		}, winFunc)

		// truncate result to start/end
		eventMask = eventMask[startIdx:endIdx]
		if err := RegisterEventSeries(strings.ReplaceAll(ev.Name, " ", "_"), eFeat, eventMask); err != nil {
			slog.Warn("unable to register event series", "error", err.Error())
			continue
		}
	}
}

func (e EventOptions) TablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(e.Events) > 0 {
		noCfg = ""
		if _, err := fmt.Fprintf(tbl, "%s%sName\tStart\tEnd\t\n", prefix, util.IndentExpand(indent, indentGrowth+1)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s%sEvents:%s\n", prefix, util.IndentExpand(indent, indentGrowth), noCfg); err != nil {
		return err
	}
	for _, ev := range e.Events {
		if _, err := fmt.Fprintf(tbl, "%s%s%s\t%s\t%s\t\n",
			prefix, util.IndentExpand(indent, indentGrowth+1),
			ev.Name, ev.Start, ev.End); err != nil {
			return err
		}
	}
	return tbl.Flush()
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
