package forecaster

import (
	"fmt"
	"io"

	"github.com/goccy/go-json"

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
	if _, err := fmt.Fprintln(w, "Series:"); err != nil {
		return err
	}
	if err := m.tablePrintSeriesOptions(w); err != nil {
		return err
	}

	if err := m.Series.TablePrint(w, "  ", "  "); err != nil {
		return err
	}
	if _, err := fmt.Fprintln(w, ""); err != nil {
		return err
	}

	if _, err := fmt.Fprintln(w, "Uncertainty:"); err != nil {
		return err
	}
	if err := m.tablePrintUncertaintyOptions(w); err != nil {
		return err
	}

	if err := m.Uncertainty.TablePrint(w, "  ", "  "); err != nil {
		return err
	}

	_, err := fmt.Fprintln(w, "")
	return err
}

func (m Model) tablePrintSeriesOptions(w io.Writer) error {
	if m.Options == nil {
		return nil
	}
	if _, err := fmt.Fprintln(w, "  Options:"); err != nil {
		return err
	}
	if m.Options.SeriesOptions == nil {
		return nil
	}
	if m.Options.SeriesOptions.OutlierOptions == nil {
		if _, err := fmt.Fprintln(w, "    Outlier Options: None"); err != nil {
			return err
		}
	} else {
		if _, err := fmt.Fprintln(w, "    Outlier Options:"); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "      Number of Passes: %d    Tukey Factor: %.3f    Lower Percentile: %.2f%%    Upper Percentile: %.2f%%\n",
			m.Options.SeriesOptions.OutlierOptions.NumPasses,
			m.Options.SeriesOptions.OutlierOptions.TukeyFactor,
			m.Options.SeriesOptions.OutlierOptions.LowerPercentile*100.0,
			m.Options.SeriesOptions.OutlierOptions.UpperPercentile*100.0,
		); err != nil {
			return err
		}
	}
	return nil
}

func (m Model) tablePrintUncertaintyOptions(w io.Writer) error {
	if m.Options == nil {
		return nil
	}
	if _, err := fmt.Fprintln(w, "  Options:"); err != nil {
		return err
	}
	if m.Options.UncertaintyOptions == nil {
		return nil
	}
	_, err := fmt.Fprintf(w, "    Residual Window: %d samples    Residual Z-Score: %.3f\n",
		m.Options.UncertaintyOptions.ResidualWindow,
		m.Options.UncertaintyOptions.ResidualZscore,
	)
	return err
}
