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

	for _, ev := range e.Events {
		if err := ev.Valid(); err != nil {
			slog.Warn("not separately modelling invalid event", "name", ev.Name, "error", err.Error())
			continue
		}

		evStd := feature.EventStandard{
			Start: ev.Start,
			End:   ev.End,
		}

		name := strings.ReplaceAll(ev.Name, " ", "_")
		feat := feature.NewEvent(name)
		if _, exists := eFeat.Get(feat); exists {
			slog.Warn("event feature already exists", "name", name)
			continue
		}
		data, err := feat.Generate(t, window, evStd)
		if err != nil {
			slog.Warn("unable to generate event feature", "error", err.Error())
			continue
		}

		eFeat.Set(feat, data)
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
