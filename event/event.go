package event

import (
	"errors"
	"time"
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
