package forecaster

import "github.com/aouyang1/go-forecaster/forecast"

// OutlierOptions configures the outlier removal pre-process using the Tukey Method. The outlier
// removal process is done by multiple iterations of fitting the training data to a model and each step
// removing outliers. For IQR set UpperPercentile too 0.75, LowerPercentile to 0.25, and TukeyFactor to 1.5.
type OutlierOptions struct {
	NumPasses       int     `json:"num_passes"`
	UpperPercentile float64 `json:"upper_percentile"`
	LowerPercentile float64 `json:"lower_percentile"`
	TukeyFactor     float64 `json:"tukey_factor"`
}

// NewOutlierOptions generates a default set of outlier options
func NewOutlierOptions() *OutlierOptions {
	return &OutlierOptions{
		NumPasses:       3,
		UpperPercentile: 0.9,
		LowerPercentile: 0.1,
		TukeyFactor:     1.0,
	}
}

type SeriesOptions struct {
	ForecastOptions *forecast.Options `json:"-"`
	OutlierOptions  *OutlierOptions   `json:"outlier_options"`
}

func NewSeriesOptions() *SeriesOptions {
	return &SeriesOptions{
		ForecastOptions: forecast.NewDefaultOptions(),
		OutlierOptions:  NewOutlierOptions(),
	}
}

type UncertaintyOptions struct {
	ForecastOptions *forecast.Options `json:"-"`
	ResidualWindow  int               `json:"residual_window"`
	ResidualZscore  float64           `json:"residual_zscore"`
}

func NewUncertaintyOptions() *UncertaintyOptions {
	return &UncertaintyOptions{
		ForecastOptions: forecast.NewDefaultOptions(),
		ResidualWindow:  100,
		ResidualZscore:  4.0,
	}
}

// Options represents all forecaster options for outlier removal, forecast fit, and uncertainty fit
type Options struct {
	SeriesOptions      *SeriesOptions      `json:"series_options"`
	UncertaintyOptions *UncertaintyOptions `json:"uncertainty_options"`
}

// NewDefaultOptions generates a default set of options for a forecaster
func NewDefaultOptions() *Options {
	return &Options{
		SeriesOptions:      NewSeriesOptions(),
		UncertaintyOptions: NewUncertaintyOptions(),
	}
}
