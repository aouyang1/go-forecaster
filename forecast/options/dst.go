package options

import (
	"log/slog"
	"time"
)

// DSTOptions lets us adjust the time to account for Daylight Saving Time behavior changes
// by timezone. In the presence of multiple timezones this will average out the effect evenly
// across the input timezones. e.g America/Los_Angeles + Europe/London will shift the time by 30min
// 2024-03-10 (America) to 2024-03-31 (Europe) and then by 60min on or after 2024-03-31.
type DSTOptions struct {
	Enabled           bool     `json:"enabled"`
	TimezoneLocations []string `json:"timezone_locations"`
}

func (d DSTOptions) AdjustTime(t []time.Time) []time.Time {
	if !d.Enabled {
		return t
	}

	offsets := loadLocationOffsets(d.TimezoneLocations)

	newT := make([]time.Time, len(t))
	for i := 0; i < len(t); i++ {
		newT[i] = adjustTime(t[i], offsets)
	}
	return newT
}

func loadLocationOffsets(names []string) []locDstOffset {
	var offsets []locDstOffset
	for _, name := range names {
		loc, err := time.LoadLocation(name)
		if err != nil {
			slog.Info("unable to load location, skipping", "location", name)
			continue
		}
		offset := getLocationDSTOffset(loc)
		offsets = append(offsets, locDstOffset{
			loc:    loc,
			offset: offset,
		})
	}
	return offsets
}

func getLocationDSTOffset(loc *time.Location) int {
	ctDec := time.Date(2024, 12, 1, 0, 0, 0, 0, loc)
	ctJul := time.Date(2024, 6, 1, 0, 0, 0, 0, loc)
	var offset int
	if ctDec.IsDST() {
		offset = deriveDSToffset(ctDec, ctJul)
	} else {
		offset = deriveDSToffset(ctJul, ctDec)
	}
	return offset
}

func deriveDSToffset(dstTime, stdTime time.Time) int {
	_, dstOffset := dstTime.Zone()
	_, stdOffset := stdTime.Zone()

	return dstOffset - stdOffset
}

type locDstOffset struct {
	loc    *time.Location
	offset int
}

// adjustTime checks a time against all dst offsets ranges and adjusts it by checking if the time is
// in a dst location offset range and finally averaging the cumulative offsets.
func adjustTime(t time.Time, offsets []locDstOffset) time.Time {
	var offsetSum int
	for _, offset := range offsets {
		locT := t.In(offset.loc)
		if locT.IsDST() {
			offsetSum += offset.offset
		}
	}
	if len(offsets) == 0 {
		return t
	}
	return t.Add(time.Duration(offsetSum/len(offsets)) * time.Second)
}
