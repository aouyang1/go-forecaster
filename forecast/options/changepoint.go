package options

import (
	"fmt"
	"io"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/util"
)

const (
	DefaultAutoNumChangepoints int = 100
)

// Changepoint describes a point in time that will change the ongoing trend. This will
// include both a bias a growth feature.
type Changepoint struct {
	T    time.Time `json:"time"`
	Name string    `json:"name"`
}

func NewChangepoint(name string, t time.Time) Changepoint {
	return Changepoint{t, name}
}

// ChangepointOptions configures the changepoint fit to either use auto-detection
// by evenly placing N changepoints in the training window. Running with auto-detection
// will generally require increasing the regularization parameter to remove changepoints
// that can cause overfitting. In general it is best to specify known changepoints which
// results in much faster training times and model size.
type ChangepointOptions struct {
	Changepoints        []Changepoint `json:"changepoints"`
	EnableGrowth        bool          `json:"enable_growth"`
	Auto                bool          `json:"auto"`
	AutoNumChangepoints int           `json:"auto_num_changepoints"`
}

func (c ChangepointOptions) TablePrint(w io.Writer, prefix, indent string, indentGrowth int) error {
	tbl := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	noCfg := " None"
	if len(c.Changepoints) > 0 {
		noCfg = ""
		fmt.Fprintf(tbl, "%s%sName\tDatetime\t\n", prefix, util.IndentExpand(indent, indentGrowth+1))
	}
	fmt.Fprintf(w, "%s%sChangepoints:%s\n", prefix, util.IndentExpand(indent, indentGrowth), noCfg)
	for _, chpt := range c.Changepoints {
		fmt.Fprintf(tbl, "%s%s%s\t%s\t\n",
			prefix, util.IndentExpand(indent, indentGrowth+1),
			chpt.Name, chpt.T)
	}
	return tbl.Flush()
}

// NewDefaultChangepointOptions generates a set of default changepoint options
func NewDefaultChangepointOptions() ChangepointOptions {
	return ChangepointOptions{
		Auto:                false,
		AutoNumChangepoints: DefaultAutoNumChangepoints,
		Changepoints:        nil,
	}
}

func (c *ChangepointOptions) GenerateAutoChangepoints(t []time.Time) []Changepoint {
	if !c.Auto {
		return nil
	}

	if c.AutoNumChangepoints == 0 {
		c.AutoNumChangepoints = DefaultAutoNumChangepoints
	}
	n := c.AutoNumChangepoints

	var minTime, maxTime time.Time
	for _, tPnt := range t {
		if minTime.IsZero() || tPnt.Before(minTime) {
			minTime = tPnt
		}
		if maxTime.IsZero() || tPnt.After(maxTime) {
			maxTime = tPnt
		}
	}

	window := maxTime.Sub(minTime)
	changepointWinNs := int64(window.Nanoseconds()) / int64(n)
	chpts := make([]Changepoint, 0, n)

	for i := 0; i < n; i++ {
		chpntTime := minTime.Add(time.Duration(changepointWinNs * int64(i)))
		chpts = append(
			chpts,
			NewChangepoint("auto_"+strconv.Itoa(i), chpntTime),
		)
	}

	// replace existing changepoints
	c.Changepoints = chpts
	return chpts
}

func (c ChangepointOptions) GenerateFeatures(t []time.Time, trainingEndTime time.Time) *feature.Set {
	chpts := c.Changepoints
	filteredChpts := make([]Changepoint, 0, len(chpts))
	for _, chpt := range chpts {
		// skip over changepoints that are after the training end time since they'll
		// not have been modeled producing zeroes in the feature set
		if chpt.T.After(trainingEndTime) {
			continue
		}
		filteredChpts = append(filteredChpts, chpt)
	}

	// create a slice of features where it goes in the order of bias, slope for each changepoint
	chptBiasFeatures := make([][]float64, len(filteredChpts))
	var chptGrowthFeatures [][]float64
	if c.EnableGrowth {
		chptGrowthFeatures = make([][]float64, len(filteredChpts))
	}
	for i := 0; i < len(filteredChpts); i++ {
		chptBias := make([]float64, len(t))
		chptBiasFeatures[i] = chptBias

		if c.EnableGrowth {
			chptGrowth := make([]float64, len(t))
			chptGrowthFeatures[i] = chptGrowth
		}

	}

	// compute dt between training end time and changepoint time
	deltaT := make([]float64, len(filteredChpts))
	for i, chpt := range filteredChpts {
		deltaT[i] = trainingEndTime.Sub(chpt.T).Seconds()
	}

	bias := 1.0
	var slope float64
	for i := 0; i < len(t); i++ {
		for j := 0; j < len(filteredChpts); j++ {
			if t[i].Equal(filteredChpts[j].T) || t[i].After(filteredChpts[j].T) {
				chptBiasFeatures[j][i] = bias

				if c.EnableGrowth {
					slope = t[i].Sub(filteredChpts[j].T).Seconds() / deltaT[j]
					chptGrowthFeatures[j][i] = slope
				}
			}
		}
	}

	feat := feature.NewSet()
	for i := 0; i < len(filteredChpts); i++ {
		chpntName := strconv.Itoa(i)
		if filteredChpts[i].Name != "" {
			chpntName = filteredChpts[i].Name
		}
		chpntBias := feature.NewChangepoint(chpntName, feature.ChangepointCompBias)
		feat.Set(chpntBias, chptBiasFeatures[i])

		if c.EnableGrowth {
			chpntGrowth := feature.NewChangepoint(chpntName, feature.ChangepointCompSlope)
			feat.Set(chpntGrowth, chptGrowthFeatures[i])
		}
	}
	return feat
}