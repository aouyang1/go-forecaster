package forecaster

import (
	"fmt"
	"os"
	"runtime/debug"
	"testing"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/timedataset"
)

func generateExampleSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 28 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateWaveY(t, 23.4, period, 7.0, 6.0*60*60+period/2.0/2.0/3.0).MaskWithTimeRange(t[minutes*4/16], t[minutes*5/16], t)).
		Add(timedataset.GenerateWaveY(t, -7.3, period, 3.0, 2*60*60+period/2.0/2.0/3.0).MaskWithWeekend(t)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes/2], 10.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes*19/20], -70.0, 0.0)).
		SetConst(t, 2.7, t[minutes/3], t[minutes/3+minutes/20]).
		SetConst(t, 175.7, t[minutes*2/3], t[minutes*2/3+minutes/80])

	return t, y
}

func runForecastExample(opt *Options, t []time.Time, y []float64, filename string) error {
	f, err := New(opt)
	if err != nil {
		return err
	}
	if err := f.Fit(t, y); err != nil {
		return err
	}

	m, err := f.Model()
	if err != nil {
		return err
	}
	if err := m.TablePrint(os.Stderr); err != nil {
		return err
	}

	file, err := os.Create(filename)
	if err != nil {
		return err
	}

	return f.PlotFit(file, nil)
}

func recoverForecastPanic(t *testing.T) {
	if r := recover(); r != nil {
		if t != nil {
			t.Errorf("panic: %v\n", r)
		} else {
			fmt.Printf("panic: %v\n", r)
		}
		debug.PrintStack()
	}
}

func setupWithOutliers() ([]time.Time, []float64, *Options) {
	t, y := generateExampleSeries()

	changepoints := []options.Changepoint{
		options.NewChangepoint("anomaly1", t[len(t)/2]),
		options.NewChangepoint("anomaly2", t[len(t)*19/20]),
	}
	events := []options.Event{
		options.NewEvent("custom_event", t[len(t)*4/16], t[len(t)*5/16]),
	}

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
				DSTOptions: options.DSTOptions{
					Enabled: false,
				},
			},
			OutlierOptions: &OutlierOptions{
				NumPasses:       3,
				TukeyFactor:     1.5,
				LowerPercentile: 0.25,
				UpperPercentile: 0.75,
			},
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
				DSTOptions: options.DSTOptions{
					Enabled: false,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
		MinUncertaintyValue: 13.0,
	}

	return t, y, opt
}

func Example_forecasterWithOutliers() {
	t, y, opt := setupWithOutliers()

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster.html"); err != nil {
		panic(err)
	}
	// Output:
}

func Example_forecasterWithOutliersConstantUncertainty() {
	t, y, opt := setupWithOutliers()
	opt.UncertaintyOptions.Percentage = 0.10
	opt.MinUncertaintyValue = 0.0

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_constant_uncertainty.html"); err != nil {
		panic(err)
	}
	// Output:
}

func generateExampleSeriesWithTrend() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 4 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes/2], 0.0, 40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes()))).
		Add(timedataset.GenerateChange(t, t[minutes*17/20], -40.0, -40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes())))

	return t, y
}

func generateExampleSeriesWithGlobalTrend() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 14 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[0], 0.0, 40.0/(t[minutes*17/20].Sub(t[minutes/2]).Minutes())))

	return t, y
}

/*
func Example_forecasterAutoChangepoint() {
	t, y := generateExampleSeries()

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Auto:                true,
					AutoNumChangepoints: 100,
					EnableGrowth:        false,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Auto:         false,
					Changepoints: []options.Changepoint{},
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}
	opt.SetMinValue(0.0)
	opt.SetMaxValue(170.0)

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_auto_changepoint.html"); err != nil {
		panic(err)
	}
	// Output:
}
*/

func Example_forecasterWithTrend() {
	t, y := generateExampleSeriesWithTrend()

	changepoints := []options.Changepoint{
		options.NewChangepoint("trendstart", t[len(t)/2]),
		options.NewChangepoint("rebaseline", t[len(t)*17/20]),
	}

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
					EnableGrowth: true,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_with_trend.html"); err != nil {
		panic(err)
	}
	// Output:
}

func Example_forecasterWithGlobalTrend() {
	t, y := generateExampleSeriesWithGlobalTrend()

	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				GrowthType: feature.GrowthLinear,
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_with_global_trend.html"); err != nil {
		panic(err)
	}
	// Output:
}

func generateExampleSeriesWithWeekendsAndEvent() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with 4 weeks
	minutes := 28 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := 86400.0
	y.Add(timedataset.GenerateConstY(minutes, 98.3)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 1.0, 2*60*60)).
		Add(timedataset.GenerateWaveY(t, 10.5, period, 3.0, 2.0*60*60+period/2.0/2.0/3.0)).
		Add(timedataset.GenerateWaveY(t, 23.4, period, 7.0, 6.0*60*60+period/2.0/2.0/3.0).MaskWithWeekend(t)).
		Add(timedataset.GenerateChange(t, t[minutes/2], 15.0, 0.0)).
		Add(timedataset.GenerateChange(t, t[minutes/2+int(float64(minutes)*1.5/28)], -15.0, 0.0)).
		Add(timedataset.GenerateNoise(t, 3.2, 3.2, period, 5.0, 0.0))

	return t, y
}

func setupWithWeekendsAndEvent() ([]time.Time, []float64, *Options) {
	t, y := generateExampleSeriesWithWeekendsAndEvent()

	minutes := 28 * 24 * 60
	eventStart := t[len(t)/2]
	eventEnd := t[len(t)/2+int(float64(minutes)*1.5/28)]
	events := []options.Event{
		options.NewEvent("special_promotion", eventStart, eventEnd),
	}

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
				DSTOptions: options.DSTOptions{
					Enabled: false,
				},
			},
			OutlierOptions: &OutlierOptions{
				NumPasses:       3,
				TukeyFactor:     1.5,
				LowerPercentile: 0.25,
				UpperPercentile: 0.75,
			},
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: true,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
				DSTOptions: options.DSTOptions{
					Enabled: false,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	return t, y, opt
}

func generateExamplePulseSeries() ([]time.Time, []float64) {
	// create a daily sine wave at minutely with one week
	minutes := 28 * 24 * 60
	t := timedataset.GenerateT(minutes, time.Minute, time.Now)
	y := make(timedataset.Series, minutes)

	period := (3 * time.Hour).Seconds()
	y.Add(timedataset.GenerateConstY(len(t), 30)).
		Add(timedataset.GeneratePulseY(t, 100.0, period, 1.0, 0.0, 0.05)).
		Add(timedataset.GenerateNoise(t, 3.2, 0.0, period, 5.0, 0.0))

	return t, y
}

func setupWithPulses() ([]time.Time, []float64, *Options) {
	t, y := generateExamplePulseSeries()

	changepoints := []options.Changepoint{}
	events := []options.Event{}

	regularization := []float64{0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0}
	opt := &Options{
		SeriesOptions: &SeriesOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewSeasonalityConfig("pulse", 3*time.Hour, 24),
					},
				},
				Iterations: 500,
				Tolerance:  1e-3,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: changepoints,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: false,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			OutlierOptions: NewOutlierOptions(),
		},
		UncertaintyOptions: &UncertaintyOptions{
			ForecastOptions: &options.Options{
				Regularization: regularization,
				SeasonalityOptions: options.SeasonalityOptions{
					SeasonalityConfigs: []options.SeasonalityConfig{
						options.NewDailySeasonalityConfig(12),
						options.NewWeeklySeasonalityConfig(12),
					},
				},
				Iterations: 250,
				Tolerance:  1e-2,
				ChangepointOptions: options.ChangepointOptions{
					Changepoints: nil,
				},
				WeekendOptions: options.WeekendOptions{
					Enabled: false,
				},
				EventOptions: options.EventOptions{
					Events: events,
				},
			},
			ResidualWindow: 100,
			ResidualZscore: 4.0,
		},
	}

	return t, y, opt
}

func Example_forecasterWithPulses() {
	t, y, opt := setupWithPulses()

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_pulses.html"); err != nil {
		panic(err)
	}
	// Output:
}

func Example_forecasterWithWeekendsAndCustomEvent() {
	t, y, opt := setupWithWeekendsAndEvent()

	defer recoverForecastPanic(nil)

	if err := runForecastExample(opt, t, y, "examples/forecaster_weekends_and_event.html"); err != nil {
		panic(err)
	}
	// Output:
}
