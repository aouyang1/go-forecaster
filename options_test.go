package forecaster

import (
	"testing"

	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/stretchr/testify/assert"
)

func TestOptionsValidate(t *testing.T) {
	testData := map[string]struct {
		opt *Options
		err error
	}{
		"nil":     {nil, nil},
		"default": {NewDefaultOptions(), nil},
		"valid min and max": {
			func() *Options {
				o := NewDefaultOptions()
				o.SetMinValue(0.0)
				o.SetMaxValue(10.0)
				return o
			}(),
			nil,
		},
		"nil series options": {
			&Options{UncertaintyOptions: NewUncertaintyOptions()},
			ErrNoSeriesOrUncertaintyOptions,
		},
		"nil uncertainty options": {
			&Options{SeriesOptions: NewSeriesOptions()},
			ErrNoSeriesOrUncertaintyOptions,
		},
		"min greater than max": {
			func() *Options {
				o := NewDefaultOptions()
				o.SetMinValue(10.0)
				o.SetMaxValue(0.0)
				return o
			}(),
			ErrMinValueGreaterThanMaxValue,
		},
		"negative min uncertainty value": {
			func() *Options {
				o := NewDefaultOptions()
				o.MinUncertaintyValue = -1.0
				return o
			}(),
			ErrNegativeMinUncertaintyValue,
		},
		"negative min uncertainty percentage": {
			func() *Options {
				o := NewDefaultOptions()
				o.MinUncertaintyPercentage = -0.1
				return o
			}(),
			ErrNegativeMinUncertaintyPercentage,
		},
		"negative outlier num passes": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.OutlierOptions.NumPasses = -1
				return o
			}(),
			ErrNegativeNumPasses,
		},
		"negative outlier tukey factor": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.OutlierOptions.TukeyFactor = -1.0
				return o
			}(),
			ErrNegativeTukeyFactor,
		},
		"outlier percentile out of range": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.OutlierOptions.LowerPercentile = -0.1
				return o
			}(),
			ErrInvalidOutlierPercentile,
		},
		"outlier lower not less than upper": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.OutlierOptions.LowerPercentile = 0.9
				o.SeriesOptions.OutlierOptions.UpperPercentile = 0.1
				return o
			}(),
			ErrLowerPercentileGreaterOrEqual,
		},
		"negative residual window": {
			func() *Options {
				o := NewDefaultOptions()
				o.UncertaintyOptions.ResidualWindow = -1
				return o
			}(),
			ErrNegativeResidualWindow,
		},
		"negative residual z-score": {
			func() *Options {
				o := NewDefaultOptions()
				o.UncertaintyOptions.ResidualZscore = -1.0
				return o
			}(),
			ErrNegativeResidualZscore,
		},
		"negative uncertainty percentage": {
			func() *Options {
				o := NewDefaultOptions()
				o.UncertaintyOptions.Percentage = -0.1
				return o
			}(),
			ErrNegativeUncertaintyPercentage,
		},
		"greater than one uncertainty percentage allowed": {
			func() *Options {
				o := NewDefaultOptions()
				o.UncertaintyOptions.Percentage = 1.5
				o.MinUncertaintyPercentage = 2.0
				return o
			}(),
			nil,
		},
		"negative regularization": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.ForecastOptions.Regularization = []float64{-1.0}
				return o
			}(),
			options.ErrNegativeRegularization,
		},
		"negative iterations": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.ForecastOptions.Iterations = -1
				return o
			}(),
			options.ErrNegativeIterations,
		},
		"negative tolerance": {
			func() *Options {
				o := NewDefaultOptions()
				o.UncertaintyOptions.ForecastOptions.Tolerance = -1.0
				return o
			}(),
			options.ErrNegativeTolerance,
		},
		"negative parallelization": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.ForecastOptions.Parallelization = -1
				return o
			}(),
			options.ErrNegativeParallelization,
		},
		"negative auto num changepoints": {
			func() *Options {
				o := NewDefaultOptions()
				o.SeriesOptions.ForecastOptions.ChangepointOptions.AutoNumChangepoints = -1
				return o
			}(),
			options.ErrNegativeAutoNumChangepoints,
		},
	}

	for name, td := range testData {
		t.Run(name, func(t *testing.T) {
			err := td.opt.Validate()
			if td.err != nil {
				assert.ErrorIs(t, err, td.err)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestNewInvalidOptions(t *testing.T) {
	opt := NewDefaultOptions()
	opt.UncertaintyOptions.ResidualWindow = -1

	f, err := New(opt)
	assert.ErrorIs(t, err, ErrNegativeResidualWindow)
	assert.Nil(t, f)
}

func TestNewNilSeriesOptions(t *testing.T) {
	f, err := New(&Options{})
	assert.ErrorIs(t, err, ErrNoSeriesOrUncertaintyOptions)
	assert.Nil(t, f)
}

func TestNewFromModelInvalidOptions(t *testing.T) {
	opt := NewDefaultOptions()
	opt.MinUncertaintyValue = -1.0

	f, err := NewFromModel(Model{Options: opt})
	assert.ErrorIs(t, err, ErrNegativeMinUncertaintyValue)
	assert.Nil(t, f)
}
