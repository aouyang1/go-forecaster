package forecast

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/forecast/options"
	"github.com/aouyang1/go-forecaster/forecast/util"
	"github.com/aouyang1/go-forecaster/models"
	"github.com/aouyang1/go-forecaster/timedataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

var (
	ErrUninitializedForecast    = errors.New("uninitialized forecast")
	ErrInsufficientTrainingData = errors.New("insufficient training data after removing Nans")
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrNoModelCoefficients      = errors.New("no model coefficients from fit")
	ErrUntrainedForecast        = errors.New("forecast has not been trained yet")
	ErrNoOptionsInModel         = errors.New("no options set in model")
	ErrNoFeaturesForFit         = errors.New("no features for fitting")
)

// Forecast represents a single forecast model of a time series. This is a linear model using
// coordinate descent to calculate the weights. This will decompose the series into an intercept,
// trend components (based on changepoint times), and seasonal components.
type Forecast struct {
	opt    *options.Options
	scores *Scores // score calculations after training

	// model coefficients
	trainEndTime    time.Time
	residual        []float64
	trainComponents Components

	featureWeights []FeatureWeight
	intercept      float64
	trained        bool
}

// New creates a new forecast instance withh thhe given options. If none are provided, a default
// is used
func New(opt *options.Options) (*Forecast, error) {
	if opt == nil {
		opt = options.NewDefaultOptions()
	}

	return &Forecast{opt: opt}, nil
}

// NewFromModel creates a new forecast instance given a forecast Model to initialize. This
// instance can be used for inference immediately and does not need to be trained again.
func NewFromModel(model Model) (*Forecast, error) {
	if model.Options == nil {
		return nil, ErrNoOptionsInModel
	}
	f := &Forecast{
		opt:            model.Options,
		trainEndTime:   model.TrainEndTime,
		intercept:      model.Weights.Intercept,
		featureWeights: model.Weights.Coef,
		scores:         model.Scores,
		trained:        true,
	}
	return f, nil
}

func (f *Forecast) generateFeatures(t []time.Time) (*feature.Set, error) {
	if f == nil {
		return nil, ErrUninitializedForecast
	}

	t = f.opt.DSTOptions.AdjustTime(t)

	tFeat, eFeat := f.opt.GenerateTimeFeatures(t)

	feat, err := f.opt.GenerateFourierFeatures(tFeat)
	if err != nil {
		return nil, err
	}
	feat.Update(eFeat)

	// do not include weekly fourier features if time range is less than 1 week
	if !f.trained && t[len(t)-1].Sub(t[0]) < time.Duration(7*24*time.Hour) {
		for _, f := range feat.Labels() {
			if val, _ := f.Get("name"); val == options.LabelSeasWeekly {
				feat.Del(f)
			}
		}
	}

	// generate changepoint features
	if !f.trained {
		f.opt.ChangepointOptions.GenerateAutoChangepoints(t)
	}
	chptFeat := f.opt.ChangepointOptions.GenerateFeatures(t, f.trainEndTime)
	feat.Update(chptFeat)

	feat.RemoveZeroOnlyFeatures()

	if !f.trained {
		return feat, nil
	}

	// evict any features that are not in the model if already trained since this is used for prediction
	relevantFeatures := make(map[string]struct{})
	for _, fw := range f.featureWeights {
		f, err := fw.ToFeature()
		if err != nil {
			return nil, fmt.Errorf("unable to extract feature from feature weight for extracting relevant features, %v, %w", fw, err)
		}
		relevantFeatures[f.String()] = struct{}{}
	}

	for _, f := range feat.Labels() {
		if _, exists := relevantFeatures[f.String()]; !exists {
			feat.Del(f)
		}
	}

	return feat, nil
}

// Fit takes the input training data and fits a forecast model for possible changepoints,
// seasonal components, and intercept
func (f *Forecast) Fit(t []time.Time, y []float64) error {
	if f == nil {
		return ErrUninitializedForecast
	}

	trainingData, err := timedataset.NewUnivariateDataset(t, y)
	if err != nil {
		return err
	}

	// remove any NaNs from training set
	trainingDataFiltered := trainingData.DropNan()
	trainingT := trainingDataFiltered.T
	if len(trainingT) <= 1 {
		return ErrInsufficientTrainingData
	}

	f.trainEndTime = timedataset.TimeSlice(trainingT).EndTime()
	// generate features
	x, err := f.generateFeatures(trainingT)
	if err != nil {
		return err
	}
	if x.Len() == 0 {
		return ErrNoFeaturesForFit
	}

	trainingY := trainingDataFiltered.Y
	if f.opt.UseLog {
		util.SliceMap(trainingY, math.Log1p)
	}
	features := x.Matrix(true)
	target := mat.NewDense(len(trainingY), 1, trainingY)

	// run coordinate descent
	lassoOpt := f.opt.NewLassoAutoOptions()
	model, err := models.NewLassoAutoRegression(lassoOpt)
	if err != nil {
		return err
	}
	if err := model.Fit(features, target); err != nil {
		return err
	}
	coef := model.Coef()
	intercept := 0.0
	if len(coef) > 0 {
		intercept = coef[0]
	}
	if len(coef) > 1 {
		coef = coef[1:]
	}
	f.trained = true

	f.intercept = intercept

	relevantFws, relevantChpts, err := f.pruneDegenerateFeatures(x.Labels(), coef)
	if err != nil {
		return err
	}
	f.featureWeights = relevantFws
	f.opt.ChangepointOptions.Changepoints = relevantChpts

	// use input training to include NaNs
	predicted, comp, err := f.Predict(trainingData.T)
	if err != nil {
		return err
	}
	f.trainComponents = comp

	scores, err := NewScores(predicted, trainingData.Y)
	if err != nil {
		return err
	}
	f.scores = scores

	residual := make([]float64, len(trainingData.T))
	floats.Add(residual, trainingData.Y)
	floats.Sub(residual, predicted)
	floats.Scale(-1.0, residual)
	f.residual = residual

	return nil
}

// pruneDegenerateFeatures removes any feature weights that are exactly equal to 0. This can happen if the LASSO
// regression regularization is strong enough to bring some of the feature weights to exactly 0.
func (f *Forecast) pruneDegenerateFeatures(labels []feature.Feature, coef []float64) ([]FeatureWeight, []options.Changepoint, error) {
	fws := make([]FeatureWeight, 0, len(coef))
	for i, c := range coef {
		fw := FeatureWeight{
			Labels: labels[i].Decode(),
			Type:   labels[i].Type(),
			Value:  c,
		}
		fws = append(fws, fw)
	}

	relevantFws := make([]FeatureWeight, 0, len(fws))
	relevantChptMap := make(map[string]struct{})
	for _, fw := range fws {
		f, err := fw.ToFeature()
		if err != nil {
			return nil, nil, fmt.Errorf("unable to extract feature to prune degenerate features, %v, %w", fw, err)
		}

		if fw.Value == 0 {
			continue
		}

		switch f.Type() {
		case feature.FeatureTypeChangepoint:
			name, exists := f.Get("name")
			if exists {
				relevantChptMap[name] = struct{}{}
			}
		}

		relevantFws = append(relevantFws, fw)
	}

	relevantChpts := make([]options.Changepoint, 0, len(f.opt.ChangepointOptions.Changepoints))
	for _, chpt := range f.opt.ChangepointOptions.Changepoints {
		_, exists := relevantChptMap[chpt.Name]
		if !exists {
			continue
		}
		relevantChpts = append(relevantChpts, chpt)
	}

	return relevantFws, relevantChpts, nil
}

// Predict takes a slice of times in any order and produces the predicted value for those
// times given a pre-trained model.
func (f *Forecast) Predict(t []time.Time) ([]float64, Components, error) {
	if f == nil {
		return nil, Components{}, ErrUninitializedForecast
	}

	if !f.trained {
		return nil, Components{}, ErrUntrainedForecast
	}

	// generate features
	x, err := f.generateFeatures(t)
	if err != nil {
		return nil, Components{}, err
	}

	changepointFeatureSet := feature.NewSet()
	seasonalityFeatureSet := feature.NewSet()
	eventFeatureSet := feature.NewSet()
	for _, feat := range x.Labels() {
		data, exists := x.Get(feat)
		if !exists {
			continue
		}
		switch feat.Type() {
		case feature.FeatureTypeChangepoint:
			changepointFeatureSet.Set(feat, data)
		case feature.FeatureTypeSeasonality:
			seasonalityFeatureSet.Set(feat, data)
		case feature.FeatureTypeEvent:
			eventFeatureSet.Set(feat, data)
		}
	}

	trendComp, err := f.runInference(changepointFeatureSet, true, len(t))
	if err != nil {
		return nil, Components{}, fmt.Errorf("unable to run inference for trend, %w", err)
	}
	seasonalityComp, err := f.runInference(seasonalityFeatureSet, false, len(t))
	if err != nil {
		return nil, Components{}, fmt.Errorf("unable to run inference for seasonality, %w", err)
	}
	eventComp, err := f.runInference(eventFeatureSet, false, len(t))
	if err != nil {
		return nil, Components{}, fmt.Errorf("unable to run inference for event, %w", err)
	}

	comp := Components{
		Trend:       trendComp,
		Seasonality: seasonalityComp,
		Event:       eventComp,
	}

	res, err := f.runInference(x, true, len(t))
	return res, comp, err
}

func (f *Forecast) runInference(x *feature.Set, withIntercept bool, numObs int) ([]float64, error) {
	if f == nil {
		return nil, nil
	}

	if x == nil {
		return nil, nil
	}

	n := x.Len()

	// account for case where we have just a bias
	if n == 0 {
		res := make([]float64, numObs)
		if withIntercept {
			floats.AddConst(f.intercept, res)
		}
		if f.opt.UseLog {
			util.SliceMap(res, math.Expm1)
		}
		return res, nil
	}

	if withIntercept {
		n += 1
	}

	xWeights := make([]float64, 0, n)
	if withIntercept {
		xWeights = append(xWeights, f.intercept)
	}

	labels := make(map[string]struct{})
	for _, f := range x.Labels() {
		labels[f.String()] = struct{}{}
	}
	for _, fw := range f.featureWeights {
		f, err := fw.ToFeature()
		if err != nil {
			return nil, fmt.Errorf("unable to convert to feature for inference, %v, %w", fw, err)
		}
		if _, exists := labels[f.String()]; exists {
			xWeights = append(xWeights, fw.Value)
		}
	}

	wMx := mat.NewDense(1, n, xWeights)
	featMx := x.Matrix(withIntercept).T()

	var resMx mat.Dense
	resMx.Mul(wMx, featMx)

	yhat := mat.Row(nil, 0, &resMx)
	if f.opt.UseLog {
		util.SliceMap(yhat, math.Expm1)
	}
	return yhat, nil
}

// Score computes the coefficient of determination of the prediction
func (f *Forecast) Score(x []time.Time, y []float64) (float64, error) {
	if x == nil {
		return 0.0, fmt.Errorf("no time slice for inference, %w", models.ErrNoDesignMatrix)
	}
	if y == nil {
		return 0.0, fmt.Errorf("no expected values for inference, %w", models.ErrNoTargetMatrix)
	}

	m := len(x)

	ym := len(y)
	if m != ym {
		return 0.0, fmt.Errorf("design matrix has %d rows and target has %d rows, %w", m, ym, models.ErrTargetLenMismatch)
	}

	res, _, err := f.Predict(x)
	if err != nil {
		return 0.0, err
	}

	return stat.RSquaredFrom(res, y, nil), nil
}

// FeatureLabels returns the slice of feature labels in the order of the coefficients
func (f *Forecast) FeatureLabels() ([]feature.Feature, error) {
	if f == nil {
		return nil, nil
	}

	fLabels := make([]feature.Feature, 0, len(f.featureWeights))
	for _, fw := range f.featureWeights {
		fl, err := fw.ToFeature()
		if err != nil {
			return nil, fmt.Errorf("unable to convert to feature in retrieving feature labels, %v, %w", fw, err)
		}
		fLabels = append(fLabels, fl)
	}
	return fLabels, nil
}

// Coefficients returns a forecast model map of coefficients keyed by the string
// representation of each feature label
func (f *Forecast) Coefficients() (map[string]float64, error) {
	if f == nil {
		return nil, ErrUninitializedForecast
	}

	if len(f.featureWeights) == 0 {
		return nil, ErrNoModelCoefficients
	}
	coef := make(map[string]float64)
	for _, fw := range f.featureWeights {
		f, err := fw.ToFeature()
		if err != nil {
			return nil, fmt.Errorf("unable to convert to feature in retrieving coefficients, %v, %w", fw, err)
		}
		coef[f.String()] = fw.Value
	}
	return coef, nil
}

// Intercept returns the intercept of the forecast model
func (f *Forecast) Intercept() float64 {
	if f == nil {
		return 0
	}
	return f.intercept
}

// Model returns the serializeable format of the forecast model composing of the
// forecast options, intercept, coefficients with their feature labels, and the
// model fit scores
func (f *Forecast) Model() (Model, error) {
	if f == nil {
		return Model{}, ErrUninitializedForecast
	}
	if !f.trained {
		return Model{}, ErrUntrainedForecast
	}

	m := Model{
		TrainEndTime: f.trainEndTime,
		Options:      f.opt,
		Weights: Weights{
			Intercept: f.intercept,
			Coef:      f.featureWeights,
		},
		Scores: f.scores,
	}
	return m, nil
}

// ModelEq returns a string representation of the model linear equation in the format of
// y ~ b + m1x1 + m2x2 + ...
func (f *Forecast) ModelEq() (string, error) {
	if f == nil {
		return "", ErrUninitializedForecast
	}

	eq := "y ~ "

	eq += fmt.Sprintf("%.2f", f.Intercept())
	for _, fw := range f.featureWeights {
		if fw.Value == 0 {
			continue
		}
		f, err := fw.ToFeature()
		if err != nil {
			return "", fmt.Errorf("unable to convert to feature generating model equation, %v, %w", fw, err)
		}
		eq += fmt.Sprintf("+%.2f*%s", fw.Value, f.String())
	}
	return eq, nil
}

// Scores returns the fit scores for evaluating how well the resulting model
// fit the training data
func (f *Forecast) Scores() Scores {
	if f == nil {
		return Scores{}
	}
	if f.scores == nil {
		return Scores{}
	}
	return *f.scores
}

// Residuals returns a slice of values representing the difference between the
// training data and the fit data
func (f *Forecast) Residuals() []float64 {
	if f == nil {
		return nil
	}
	res := make([]float64, len(f.residual))
	copy(res, f.residual)
	return res
}

// TrendComponent represents the overall trend component of the model which is determined
// by the changepoints.
func (f *Forecast) TrendComponent() []float64 {
	if f == nil {
		return nil
	}
	res := make([]float64, len(f.trainComponents.Trend))
	copy(res, f.trainComponents.Trend)
	return res
}

// SeasonalityComponent represents the overall seasonal component of the model
func (f *Forecast) SeasonalityComponent() []float64 {
	if f == nil {
		return nil
	}
	res := make([]float64, len(f.trainComponents.Seasonality))
	copy(res, f.trainComponents.Seasonality)
	return res
}

// EventComponent represents the overall event components in the model
func (f *Forecast) EventComponent() []float64 {
	if f == nil {
		return nil
	}
	res := make([]float64, len(f.trainComponents.Event))
	copy(res, f.trainComponents.Event)
	return res
}
