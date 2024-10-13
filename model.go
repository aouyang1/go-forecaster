package forecaster

import (
	"encoding/json"
	"fmt"
	"io"
	"text/tabwriter"

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
	fmt.Fprintln(w, "  Options:")
	fmt.Fprintf(w, "    Training End Time: %s\n", m.Series.TrainEndTime)
	fmt.Fprintf(w, "    Daily Orders: %d    Weekly Orders: %d    Regularization: %.3f\n",
		m.Series.Options.DailyOrders,
		m.Series.Options.WeeklyOrders,
		m.Series.Options.Regularization,
	)
	if m.Options.SeriesOptions.OutlierOptions == nil {
		fmt.Fprintln(w, "    Outlier Options: N/A")
	} else {
		fmt.Fprintln(w, "    Outlier Options:")
		fmt.Fprintf(w, "      Number of Passes: %d    Tukey Factor: %.3f    Lower Percentile: %.2f%%    Upper Percentile: %.2f%%\n",
			m.Options.SeriesOptions.OutlierOptions.NumPasses,
			m.Options.SeriesOptions.OutlierOptions.TukeyFactor,
			m.Options.SeriesOptions.OutlierOptions.LowerPercentile*100.0,
			m.Options.SeriesOptions.OutlierOptions.UpperPercentile*100.0,
		)
	}

	if len(m.Series.Options.ChangepointOptions.Changepoints) == 0 {
		fmt.Fprintln(w, "    Changepoints: None")
	} else {
		fmt.Fprintln(w, "    Changepoints:")
		for _, chpt := range m.Series.Options.ChangepointOptions.Changepoints {
			fmt.Fprintf(w, "      %s: %s\n", chpt.Name, chpt.T)
		}
	}

	fmt.Fprintln(w, "  Scores:")
	fmt.Fprintf(w, "       MAPE: %.3f    MSE: %.3f    R2: %.3f\n",
		m.Series.Scores.MAPE,
		m.Series.Scores.MSE,
		m.Series.Scores.R2,
	)

	fmt.Fprintln(w, "  Weights:")
	tblSeriesWeights := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tblSeriesWeights, "Type\tLabels\tValue\t")
	fmt.Fprintf(tblSeriesWeights, "Intercept\t\t%.5f\t\n", m.Series.Weights.Intercept)
	for _, fw := range m.Series.Weights.Coef {
		labelOut, err := json.Marshal(fw.Labels)
		if err != nil {
			return err
		}
		val := fmt.Sprintf("%.3f", fw.Value)
		if fw.Value == 0 {
			val = "..."
		}
		fmt.Fprintf(tblSeriesWeights, "%s\t%s\t%s\t\n", fw.Type, string(labelOut), val)
	}
	tblSeriesWeights.Flush()
	fmt.Fprintln(w, "")

	fmt.Fprintln(w, "Uncertainty:")
	fmt.Fprintln(w, "  Options:")
	fmt.Fprintf(w, "    Training End Time: %s\n", m.Uncertainty.TrainEndTime)
	fmt.Fprintf(w, "    Residual Window: %d samples    Residual Z-Score: %.3f\n",
		m.Options.UncertaintyOptions.ResidualWindow,
		m.Options.UncertaintyOptions.ResidualZscore,
	)
	fmt.Fprintf(w, "    Daily Orders: %d    Weekly Orders: %d    Regularization: %.3f\n",
		m.Uncertainty.Options.DailyOrders,
		m.Uncertainty.Options.WeeklyOrders,
		m.Uncertainty.Options.Regularization,
	)
	if len(m.Series.Options.ChangepointOptions.Changepoints) == 0 {
		fmt.Fprintln(w, "    Changepoints: None")
	} else {
		fmt.Fprintln(w, "    Changepoints:")
		for _, chpt := range m.Series.Options.ChangepointOptions.Changepoints {
			fmt.Fprintf(w, "      %s: %s\n", chpt.Name, chpt.T)
		}
	}

	fmt.Fprintln(w, "  Scores:")
	fmt.Fprintf(w, "       MAPE: %.3f    MSE: %.3f    R2: %.3f\n",
		m.Uncertainty.Scores.MAPE,
		m.Uncertainty.Scores.MSE,
		m.Uncertainty.Scores.R2,
	)

	fmt.Fprintln(w, "  Weights:")
	tblUncertaintyWeights := tabwriter.NewWriter(w, 0, 0, 1, ' ', tabwriter.AlignRight)
	fmt.Fprintln(tblUncertaintyWeights, "Type\tLabels\tValue\t")
	fmt.Fprintf(tblUncertaintyWeights, "Intercept\t\t%.5f\t\n", m.Uncertainty.Weights.Intercept)
	for _, fw := range m.Uncertainty.Weights.Coef {
		labelOut, err := json.Marshal(fw.Labels)
		if err != nil {
			return err
		}
		val := fmt.Sprintf("%.3f", fw.Value)
		if fw.Value == 0 {
			val = "..."
		}
		fmt.Fprintf(tblUncertaintyWeights, "%s\t%s\t%s\t\n", fw.Type, string(labelOut), val)
	}
	tblUncertaintyWeights.Flush()
	fmt.Fprintln(w, "")
	return nil
}
