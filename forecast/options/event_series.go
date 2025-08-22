package options

import (
	"fmt"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
)

type EventSeriesOptions []EventSeries

type EventSeries interface {
	Name() string
	GenerateMask(t []time.Time, window string) ([]float64, error)
}

func RegisterEventSeries(name string, eFeat *feature.Set, mask []float64) error {
	feat := feature.NewEvent(name)
	if _, exists := eFeat.Get(feat); exists {
		return fmt.Errorf("event feature already exists, %s", name)
	}

	eFeat.Set(feat, mask)
	return nil
}
