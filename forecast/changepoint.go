package forecast

import "time"

// Changepoint describes a point in time that will change the ongoing trend. This will
// include both a bias a growth feature.
type Changepoint struct {
	T    time.Time `json:"time"`
	Name string    `json:"name"`
}

func NewChangepoint(name string, t time.Time) Changepoint {
	return Changepoint{t, name}
}
