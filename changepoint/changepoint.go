package changepoint

import "time"

type Changepoint struct {
	T    time.Time `json:"time"`
	Name string    `json:"name"`
}

func New(name string, t time.Time) Changepoint {
	return Changepoint{t, name}
}
