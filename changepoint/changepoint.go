package changepoint

import "time"

type Changepoint struct {
	T    time.Time
	Name string
}

func New(name string, t time.Time) Changepoint {
	return Changepoint{t, name}
}
