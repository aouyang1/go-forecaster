package forecast

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/forecast/util"
)

var ErrUnknownFeatureType = errors.New("unknown feature type")

// Model represents a serializeable format of a forecast storing the forecast options, fit scores,
// and coefficients
type Model struct {
	TrainEndTime time.Time        `json:"train_end_time"`
	Options      *options.Options `json:"options"`
	Scores       *Scores          `json:"scores"`
	Weights      Weights          `json:"weights"`
}

func (m Model) TablePrint(w io.Writer, prefix, indent string) error {
	fmt.Fprintf(w, "%s%sForecast:\n", prefix, util.IndentExpand(indent, 0))

	fmt.Fprintf(w, "%s%sTraining End Time: %s\n", prefix, util.IndentExpand(indent, 1), m.TrainEndTime)

	if m.Options != nil {
		fmt.Fprintf(w, "%s%sRegularization: %.3f\n", prefix, util.IndentExpand(indent, 1), m.Options.Regularization)

		if err := m.Options.SeasonalityOptions.TablePrint(w, prefix, indent, 1); err != nil {
			return err
		}

		if err := m.Options.ChangepointOptions.TablePrint(w, prefix, indent, 1); err != nil {
			return err
		}

		if !m.Options.WeekendOptions.Enabled {
			fmt.Fprintf(w, "%s%sWeekends: None\n", prefix, util.IndentExpand(indent, 1))
		} else {
			fmt.Fprintf(w, "%s%sWeekends:\n", prefix, util.IndentExpand(indent, 1))
			fmt.Fprintf(w, "%s%sBefore: %s, After: %s\n",
				prefix, util.IndentExpand(indent, 2),
				-m.Options.WeekendOptions.DurBefore, m.Options.WeekendOptions.DurAfter)
		}

		if err := m.Options.EventOptions.TablePrint(w, prefix, indent, 1); err != nil {
			return err
		}
	}

	if m.Scores != nil {
		fmt.Fprintf(w, "%s%sScores:\n", prefix, util.IndentExpand(indent, 0))
		fmt.Fprintf(w, "%s%sMAPE: %.3f    MSE: %.3f    R2: %.3f\n",
			prefix, util.IndentExpand(indent, 1),
			m.Scores.MAPE,
			m.Scores.MSE,
			m.Scores.R2,
		)
	}

	return m.Weights.tablePrint(w, prefix, indent, 0)
}

// Weights stores the coefficients for the forecast model
type Weights struct {
	Coef []FeatureWeight `json:"coefficients"`
}

// FeatureLabels returns all of the feature labels in the same order as the coefficients
func (w *Weights) FeatureLabels() ([]feature.Feature, error) {
	labels := make([]feature.Feature, 0, len(w.Coef))
	for _, fw := range w.Coef {
		feat, err := fw.ToFeature()
		if err != nil {
			return nil, err
		}
		labels = append(labels, feat)
	}
	return labels, nil
}

// Coefficients returns a slice copy of the coefficients ignoring the intercept.
func (w *Weights) Coefficients() []float64 {
	coef := make([]float64, 0, len(w.Coef))
	for _, fw := range w.Coef {
		coef = append(coef, fw.Value)
	}
	return coef
}

func (w Weights) tablePrint(wr io.Writer, prefix, indent string, indentGrowth int) error {
	fmt.Fprintf(wr, "%s%sWeights:\n", prefix, util.IndentExpand(indent, indentGrowth))
	tbl := tabwriter.NewWriter(wr, 0, 0, 1, ' ', tabwriter.AlignRight)
	fmt.Fprintf(tbl, "%s%sType\tLabels\tValue\t\n", prefix, util.IndentExpand(indent, indentGrowth+1))
	for _, fw := range w.Coef {
		labelOut, err := json.Marshal(fw.Labels)
		if err != nil {
			return err
		}
		val := fmt.Sprintf("%.3f", fw.Value)
		if fw.Value == 0 {
			val = "..."
		}
		fmt.Fprintf(tbl, "%s%s%s\t%s\t%s\t\n",
			prefix, util.IndentExpand(indent, 1),
			fw.Type, string(labelOut), val)
	}
	return tbl.Flush()
}

// FeatureWeight represents a feature described with a type e.g. changepoint, labels and the value
type FeatureWeight struct {
	Labels map[string]string   `json:"labels"`
	Type   feature.FeatureType `json:"type"`
	Value  float64             `json:"value"`
}

func NewFeatureWeight(f feature.Feature, val float64) FeatureWeight {
	return FeatureWeight{
		Labels: f.Decode(),
		Type:   f.Type(),
		Value:  val,
	}
}

// ToFeature transforms the Type and Labels into a feature type
func (fw *FeatureWeight) ToFeature() (feature.Feature, error) {
	switch fw.Type {
	case feature.FeatureTypeChangepoint:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Changepoint)
		if err := json.Unmarshal(bytes, feat); err != nil {
			return nil, err
		}
		return feat, nil

	case feature.FeatureTypeSeasonality:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Seasonality)
		if err := json.Unmarshal(bytes, &feat); err != nil {
			return nil, err
		}
		return feat, nil

	case feature.FeatureTypeEvent:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Event)
		if err := json.Unmarshal(bytes, feat); err != nil {
			return nil, err
		}
		return feat, nil

	case feature.FeatureTypeGrowth:
		bytes, err := json.Marshal(fw.Labels)
		if err != nil {
			return nil, err
		}
		feat := new(feature.Growth)
		if err := json.Unmarshal(bytes, feat); err != nil {
			return nil, err
		}
		return feat, nil
	}

	return nil, ErrUnknownFeatureType
}
