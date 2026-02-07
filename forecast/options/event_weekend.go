package options

import (
	"log/slog"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
)

// WeekendOptions lets us model weekends separately from weekdays.
type WeekendOptions struct {
	Enabled          bool          `json:"enabled"`
	TimezoneOverride string        `json:"timezone_override"`
	DurBefore        time.Duration `json:"duration_before"`
	DurAfter         time.Duration `json:"duration_after"`
}

func (w *WeekendOptions) generateEventMask(t []time.Time, eFeat *feature.Set, window string) {
	if !w.Enabled {
		return
	}
	evWknd := feature.EventWeekend{
		TimezoneOverride: w.TimezoneOverride,
		DurBefore:        w.DurBefore,
		DurAfter:         w.DurAfter,
	}
	evWknd.Validate()

	feat := feature.NewEvent(feature.EventNameWeekend)
	if _, exists := eFeat.Get(feat); exists {
		slog.Warn("event feature already exists", "name", feature.EventNameWeekend)
		return
	}

	data, err := feat.Generate(t, window, evWknd)
	if err != nil {
		slog.Warn("unable to generate event feature", "error", err.Error())
		return
	}

	eFeat.Set(feat, data)
}
