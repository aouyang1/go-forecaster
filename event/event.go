package event

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rickar/cal/v2"
	"github.com/rickar/cal/v2/us"
)

var (
	ErrStartAfterEnd = errors.New("event start time is after end time")
	ErrUnsetTime     = errors.New("unset event start or end time")
	ErrNoEventName   = errors.New("no event name")
)

// represents a time span to model separately
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
				Name:  strings.ReplaceAll(fmt.Sprintf("%s_%d", hol.Name, i), " ", "_"),
				Start: observed.Add(-durBefore),
				End:   observed.Add(24 * time.Hour).Add(durAfter),
			})
		}
	}
	return events
}
