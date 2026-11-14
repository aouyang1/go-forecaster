package forecaster

import (
	"errors"
	"fmt"
	"math"

	"github.com/aouyang1/go-forecaster/forecast/options"
)

var (
	ErrMinValueGreaterThanMaxValue      = errors.New("min value is greater than max value")
	ErrNegativeMinUncertaintyValue      = errors.New("min uncertainty value must be non-negative")
	ErrNegativeMinUncertaintyPercentage = errors.New("min uncertainty percentage must be non-negative")
	ErrNegativeNumPasses                = errors.New("outlier number of passes must be non-negative")
	ErrInvalidOutlierPercentile         = errors.New("outlier percentile must be between 0 and 1")
	ErrLowerPercentileGreaterOrEqual    = errors.New("outlier lower percentile must be less than upper percentile")
	ErrNegativeTukeyFactor              = errors.New("outlier tukey factor must be non-negative")
	ErrNegativeResidualWindow           = errors.New("residual window must be non-negative")
	ErrNegativeResidualZscore           = errors.New("residual z-score must be non-negative")
	ErrNegativeUncertaintyPercentage    = errors.New("uncertainty percentage must be non-negative")
)

// OutlierOptions configures the outlier removal pre-process using the Tukey Method. The outlier
// removal process is done by multiple iterations of fitting the training data to a model and each step
// removing outliers. For IQR set UpperPercentile too 0.75, LowerPercentile to 0.25, and TukeyFactor to 1.5.
type OutlierOptions struct {
	NumPasses       int     `json:"num_passes"`
	UpperPercentile float64 `json:"upper_percentile"`
	LowerPercentile float64 `json:"lower_percentile"`
	TukeyFactor     float64 `json:"tukey_factor"`

	// Events are explicitly set time ranges that are excluded from training
	Events []options.Event `json:"events"`
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

// Validate runs basic validation on the outlier options
func (o *OutlierOptions) Validate() error {
	if o == nil {
		return nil
	}
	if o.NumPasses < 0 {
		return ErrNegativeNumPasses
	}
	if o.TukeyFactor < 0 {
		return ErrNegativeTukeyFactor
	}
	if o.LowerPercentile < 0 || o.LowerPercentile > 1 || o.UpperPercentile < 0 || o.UpperPercentile > 1 {
		return ErrInvalidOutlierPercentile
	}
	if o.LowerPercentile >= o.UpperPercentile {
		return ErrLowerPercentileGreaterOrEqual
	}
	return nil
}

type SeriesOptions struct {
	ForecastOptions *options.Options `json:"forecast_options"`
	OutlierOptions  *OutlierOptions  `json:"outlier_options"`
}

func NewSeriesOptions() *SeriesOptions {
	return &SeriesOptions{
		ForecastOptions: options.NewDefaultOptions(),
		OutlierOptions:  NewOutlierOptions(),
	}
}

// Validate runs basic validation on the series options and its nested forecast and outlier options
func (o *SeriesOptions) Validate() error {
	if o == nil {
		return nil
	}
	if o.ForecastOptions != nil {
		if err := o.ForecastOptions.Validate(); err != nil {
			return fmt.Errorf("forecast options, %w", err)
		}
	}
	if o.OutlierOptions != nil {
		if err := o.OutlierOptions.Validate(); err != nil {
			return fmt.Errorf("outlier options, %w", err)
		}
	}
	return nil
}

type UncertaintyOptions struct {
	ForecastOptions *options.Options `json:"forecast_options"`
	ResidualWindow  int              `json:"residual_window"`
	ResidualZscore  float64          `json:"residual_zscore"`
	Percentage      float64          `json:"percentage"`
}

func NewUncertaintyOptions() *UncertaintyOptions {
	return &UncertaintyOptions{
		ForecastOptions: options.NewDefaultOptions(),
		ResidualWindow:  100,
		ResidualZscore:  4.0,
	}
}

// Validate runs basic validation on the uncertainty options and its nested forecast options
func (o *UncertaintyOptions) Validate() error {
	if o == nil {
		return nil
	}
	if o.ForecastOptions != nil {
		if err := o.ForecastOptions.Validate(); err != nil {
			return fmt.Errorf("forecast options, %w", err)
		}
	}
	if o.ResidualWindow < 0 {
		return ErrNegativeResidualWindow
	}
	if o.ResidualZscore < 0 {
		return ErrNegativeResidualZscore
	}
	if o.Percentage < 0 {
		return ErrNegativeUncertaintyPercentage
	}
	return nil
}

// Options represents all forecaster options for outlier removal, forecast fit, and uncertainty fit
type Options struct {
	SeriesOptions      *SeriesOptions      `json:"series_options"`
	UncertaintyOptions *UncertaintyOptions `json:"uncertainty_options"`
	MinValue           *float64            `json:"min_value"`
	MaxValue           *float64            `json:"max_value"`

	// both value and percentage can be used but the one with the greater uncertainty will be used
	// when predicting the uncertainty value.
	MinUncertaintyValue      float64 `json:"min_uncertainty_value"`
	MinUncertaintyPercentage float64 `json:"min_uncertainty_percentage"`
}

// NewDefaultOptions generates a default set of options for a forecaster
func NewDefaultOptions() *Options {
	return &Options{
		SeriesOptions:      NewSeriesOptions(),
		UncertaintyOptions: NewUncertaintyOptions(),
	}
}

// Validate runs basic validation on all forecaster options including the nested series and
// uncertainty options. A nil receiver is considered valid.
func (o *Options) Validate() error {
	if o == nil {
		return nil
	}
	if o.SeriesOptions == nil || o.UncertaintyOptions == nil {
		return ErrNoSeriesOrUncertaintyOptions
	}
	if err := o.SeriesOptions.Validate(); err != nil {
		return fmt.Errorf("series options, %w", err)
	}
	if err := o.UncertaintyOptions.Validate(); err != nil {
		return fmt.Errorf("uncertainty options, %w", err)
	}
	if o.MinValue != nil && o.MaxValue != nil && *o.MinValue > *o.MaxValue {
		return ErrMinValueGreaterThanMaxValue
	}
	if o.MinUncertaintyValue < 0 {
		return ErrNegativeMinUncertaintyValue
	}
	if o.MinUncertaintyPercentage < 0 {
		return ErrNegativeMinUncertaintyPercentage
	}
	return nil
}

func (o *Options) SetMinValue(val float64) {
	if math.IsNaN(val) {
		return
	}
	o.MinValue = &val
}

func (o *Options) SetMaxValue(val float64) {
	if math.IsNaN(val) {
		return
	}
	o.MaxValue = &val
}
