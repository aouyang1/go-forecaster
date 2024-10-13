package forecast

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/aouyang1/go-forecaster/feature"
	"github.com/aouyang1/go-forecaster/models"
	"github.com/aouyang1/go-forecaster/timedataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrUninitializedForecast    = errors.New("uninitialized forecast")
	ErrInsufficientTrainingData = errors.New("insufficient training data after removing Nans")
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrNoModelCoefficients      = errors.New("no model coefficients from fit")
	ErrUntrainedForecast        = errors.New("forecast has not been trained yet")
)

// Forecast represents a single forecast model of a time series. This is a linear model using
// coordinate descent to calculate the weights. This will decompose the series into an intercept,
// trend components (based on changepoint times), and seasonal components.
type Forecast struct {
	opt    *Options
	scores *Scores // score calculations after training

	// model coefficients
	fLabels *feature.Labels

	trainEndTime    time.Time
	residual        []float64
	trainComponents Components

	coef      []float64
	intercept float64
	trained   bool
}

// New creates a new forecast instance withh thhe given options. If none are provided, a default
// is used
func New(opt *Options) (*Forecast, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}

	return &Forecast{opt: opt}, nil
}

// NewFromModel creates a new forecast instance given a forecast Model to initialize. This
// instance can be used for inferrence immediately and does not need to be trained again.
func NewFromModel(model Model) (*Forecast, error) {
	fLabels, err := model.Weights.FeatureLabels()
	if err != nil {
		return nil, err
	}

	f := &Forecast{
		opt:          model.Options,
		fLabels:      fLabels,
		trainEndTime: model.TrainEndTime,
		intercept:    model.Weights.Intercept,
		coef:         model.Weights.Coefficients(),
		scores:       model.Scores,
		trained:      true,
	}
	return f, nil
}

func (f *Forecast) generateFeatures(t []time.Time) (feature.Set, error) {
	if f == nil {
		return nil, ErrUninitializedForecast
	}

	tFeat := generateTimeFeatures(t, f.opt)

	feat, err := generateFourierFeatures(tFeat, f.opt)
	if err != nil {
		return nil, err
	}

	// do not include weekly fourier features if time range is less than 1 week
	if t[len(t)-1].Sub(t[0]) < time.Duration(7*24*time.Hour) {
		for label, f := range feat {
			if val, _ := f.F.Get("name"); val == "dow" {
				delete(feat, label)
			}
		}
	}

	// generate changepoint features
	var chptFeat feature.Set
	if f.opt.ChangepointOptions.Auto && !f.trained {
		if f.opt.ChangepointOptions.AutoNumChangepoints == 0 {
			f.opt.ChangepointOptions.AutoNumChangepoints = DefaultAutoNumChangepoints
		}
		f.opt.ChangepointOptions.Changepoints = generateAutoChangepoints(t, f.opt.ChangepointOptions.AutoNumChangepoints)
	}
	chptFeat = generateChangepointFeatures(t, f.opt.ChangepointOptions.Changepoints, f.trainEndTime)
	for chptLabel, feature := range chptFeat {
		feat[chptLabel] = feature
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
	trainingT := make([]time.Time, 0, len(trainingData.T))
	trainingY := make([]float64, 0, len(trainingData.Y))

	// drop out nans
	for i := 0; i < len(trainingData.T); i++ {
		if math.IsNaN(trainingData.Y[i]) {
			continue
		}
		trainingT = append(trainingT, trainingData.T[i])
		trainingY = append(trainingY, trainingData.Y[i])
	}

	if len(trainingT) <= 1 {
		return ErrInsufficientTrainingData
	}

	f.trainEndTime = trainingT[len(trainingT)-1]
	// generate features
	x, err := f.generateFeatures(trainingT)
	if err != nil {
		return err
	}

	f.fLabels = x.Labels()

	features := x.MatrixSlice(true)
	observations := trainingY

	// run coordinate descent with lambda set too 0 which is equivalent to OLS
	lassoOpt := models.NewDefaultLassoOptions()
	lassoOpt.Lambda = f.opt.Regularization
	f.intercept, f.coef, err = models.LassoRegression(features, observations, lassoOpt)
	if err != nil {
		return err
	}
	f.trained = true

	// TODO: prune irrelevant changepoints

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

	changepointFeatureSet := make(feature.Set)
	seasonalityFeatureSet := make(feature.Set)
	for label, feat := range x {
		switch feat.F.Type() {
		case feature.FeatureTypeChangepoint:
			changepointFeatureSet[label] = feat
		case feature.FeatureTypeSeasonality:
			seasonalityFeatureSet[label] = feat
		}
	}

	comp := Components{
		Trend:       f.runInference(changepointFeatureSet, true),
		Seasonality: f.runInference(seasonalityFeatureSet, false),
	}

	res := f.runInference(x, true)
	return res, comp, nil
}

func (f *Forecast) runInference(x feature.Set, withIntercept bool) []float64 {
	if f == nil {
		return nil
	}

	if len(x) == 0 {
		return nil
	}

	xLabels := x.Labels()

	n := xLabels.Len()
	if withIntercept {
		n += 1
	}

	xWeights := make([]float64, 0, n)
	if withIntercept {
		xWeights = append(xWeights, f.intercept)
	}

	for _, xFeat := range xLabels.Labels() {
		if wIdx, exists := f.fLabels.Index(xFeat); exists {
			xWeights = append(xWeights, f.coef[wIdx])
		}
	}

	wMx := mat.NewDense(1, n, xWeights)
	featMx := x.Matrix(withIntercept).T()

	var resMx mat.Dense
	resMx.Mul(wMx, featMx)

	yhat := mat.Row(nil, 0, &resMx)

	return yhat
}

// FeatureLabels returns the slice of feature labels in the order of the coefficients
func (f *Forecast) FeatureLabels() []feature.Feature {
	if f == nil {
		return nil
	}

	return f.fLabels.Labels()
}

// Coefficients returns a forecast model map of coefficients keyed by the string
// representation of each feature label
func (f *Forecast) Coefficients() (map[string]float64, error) {
	if f == nil {
		return nil, ErrUninitializedForecast
	}

	labels := f.fLabels.Labels()
	if len(labels) == 0 || len(f.coef) == 0 {
		return nil, ErrNoModelCoefficients
	}
	coef := make(map[string]float64)
	for i := 0; i < len(f.coef); i++ {
		coef[labels[i].String()] = f.coef[i]
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

	fws := make([]FeatureWeight, 0, len(f.coef))
	labels := f.fLabels.Labels()
	for i, c := range f.coef {
		fw := FeatureWeight{
			Labels: labels[i].Decode(),
			Type:   labels[i].Type(),
			Value:  c,
		}
		fws = append(fws, fw)
	}
	w := Weights{
		Intercept: f.intercept,
		Coef:      fws,
	}
	m := Model{
		TrainEndTime: f.trainEndTime,
		Options:      f.opt,
		Weights:      w,
		Scores:       f.scores,
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

	coef, err := f.Coefficients()
	if err != nil {
		return "", err
	}

	eq += fmt.Sprintf("%.2f", f.Intercept())
	labels := f.fLabels.Labels()
	for i := 0; i < len(f.coef); i++ {
		w := coef[labels[i].String()]
		if w == 0 {
			continue
		}
		eq += fmt.Sprintf("+%.2f*%s", w, labels[i])
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
