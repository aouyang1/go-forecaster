package changepoint

import "time"

type Changepoint struct {
	T             time.Time `json:"time"`
	Name          string    `json:"name"`
	DisableGrowth bool      `json:"disable_growth"`
}

func New(name string, t time.Time, disableGrowth bool) Changepoint {
	return Changepoint{t, name, disableGrowth}
}
