package forecaster

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/aouyang1/go-forecaster/forecast"
)

// Model is a serializeable representation of the forecaster's configurations and models for the
// forecast and uncertainty.
type Model struct {
	Options     *Options       `json:"options"`
	Series      forecast.Model `json:"series_model"`
	Uncertainty forecast.Model `json:"uncertainty_model"`
}

func (m Model) JSONPrettyPrint(w io.Writer) error {
	out, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}

	_, err = fmt.Fprintln(w, string(out))
	return err
}

func (m Model) TablePrint(w io.Writer) error {
	fmt.Fprintln(w, "Series:")
	if m.Options != nil {
		fmt.Fprintln(w, "  Options:")
		if m.Options.SeriesOptions != nil {
			if m.Options.SeriesOptions.OutlierOptions == nil {
				fmt.Fprintln(w, "    Outlier Options: None")
			} else {
				fmt.Fprintln(w, "    Outlier Options:")
				fmt.Fprintf(w, "      Number of Passes: %d    Tukey Factor: %.3f    Lower Percentile: %.2f%%    Upper Percentile: %.2f%%\n",
					m.Options.SeriesOptions.OutlierOptions.NumPasses,
					m.Options.SeriesOptions.OutlierOptions.TukeyFactor,
					m.Options.SeriesOptions.OutlierOptions.LowerPercentile*100.0,
					m.Options.SeriesOptions.OutlierOptions.UpperPercentile*100.0,
				)
			}
		}
	}

	if err := m.Series.TablePrint(w, "  ", "  "); err != nil {
		return err
	}
	fmt.Fprintln(w, "")

	fmt.Fprintln(w, "Uncertainty:")
	if m.Options != nil {
		fmt.Fprintln(w, "  Options:")
		if m.Options.UncertaintyOptions != nil {
			fmt.Fprintf(w, "    Residual Window: %d samples    Residual Z-Score: %.3f\n",
				m.Options.UncertaintyOptions.ResidualWindow,
				m.Options.UncertaintyOptions.ResidualZscore,
			)
		}
	}

	if err := m.Uncertainty.TablePrint(w, "  ", "  "); err != nil {
		return err
	}

	fmt.Fprintln(w, "")
	return nil
}
