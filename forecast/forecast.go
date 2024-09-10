package forecast

import (
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/aouyang1/go-forecast/feature"
	"github.com/aouyang1/go-forecast/models"
	"github.com/aouyang1/go-forecast/timedataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrNonMontonic              = errors.New("time feature is not monotonic")
	ErrNoTrainingData           = errors.New("no training data")
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrNoModelCoefficients      = errors.New("no model coefficients from fit")
)

type Forecast struct {
	opt    *Options
	scores *Scores // score calculations after training

	// model coefficients
	fLabels *FeatureLabels

	residual    []float64
	trend       []float64
	seasonality []float64

	coef      []float64
	intercept float64
}

func New(opt *Options) (*Forecast, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}

	return &Forecast{opt: opt}, nil
}

func NewFromModel(model Model) (*Forecast, error) {
	fLabels, err := model.Weights.FeatureLabels()
	if err != nil {
		return nil, err
	}

	f := &Forecast{
		opt:       model.Options,
		fLabels:   fLabels,
		intercept: model.Weights.Intercept,
		coef:      model.Weights.Coefficients(),
	}
	return f, nil
}

func (f *Forecast) generateFeatures(t []time.Time) (FeatureSet, error) {
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
	chptFeat := generateChangepointFeatures(t, f.opt.Changepoints)
	for chptLabel, feature := range chptFeat {
		feat[chptLabel] = feature
	}

	return feat, nil
}

func (f *Forecast) Fit(trainingData *timedataset.TimeDataset) error {
	if trainingData == nil {
		return ErrNoTrainingData
	}

	// remove any NaNs from training set
	trainingT := make([]time.Time, 0, len(trainingData.T))
	trainingY := make([]float64, 0, len(trainingData.Y))

	// check for monontic timestamps
	var lastT time.Time
	for i := 0; i < len(trainingData.T); i++ {
		currT := trainingData.T[i]
		if currT.Before(lastT) || currT.Equal(lastT) {
			return fmt.Errorf("non-monotonic at %d, %w", i, ErrNonMontonic)
		}
		lastT = currT
		if math.IsNaN(trainingData.Y[i]) {
			continue
		}
		trainingT = append(trainingT, currT)
		trainingY = append(trainingY, trainingData.Y[i])
	}

	// generate features
	x, err := f.generateFeatures(trainingT)
	if err != nil {
		return err
	}

	f.fLabels = x.Labels()

	features := x.Matrix(true)
	observations := ObservationMatrix(trainingY)

	// run coordinate descent with lambda set too 0 which is equivalent to OLS
	lassoOpt := models.NewDefaultLassoOptions()
	lassoOpt.Lambda = 0.0
	f.intercept, f.coef, err = models.LassoRegression2(features, observations, lassoOpt)
	if err != nil {
		return err
	}

	// use input training to include NaNs
	predicted, err := f.Predict(trainingData.T)
	if err != nil {
		return err
	}
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

	// compute changepoint and seasonal components
	x, err = f.generateFeatures(trainingData.T)
	if err != nil {
		return err
	}

	changepointFeatureSet := make(FeatureSet)
	seasonalityFeatureSet := make(FeatureSet)
	for label, feat := range x {
		switch feat.F.Type() {
		case feature.FeatureTypeChangepoint:
			changepointFeatureSet[label] = feat
		case feature.FeatureTypeSeasonality:
			seasonalityFeatureSet[label] = feat
		}
	}

	f.trend = f.runInference(changepointFeatureSet, true)
	f.seasonality = f.runInference(seasonalityFeatureSet, false)
	return nil
}

func (f *Forecast) Predict(t []time.Time) ([]float64, error) {
	// generate features
	x, err := f.generateFeatures(t)
	if err != nil {
		return nil, err
	}

	res := f.runInference(x, true)
	return res, nil
}

func (f *Forecast) runInference(x FeatureSet, withIntercept bool) []float64 {
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

func (f *Forecast) FeatureLabels() []feature.Feature {
	return f.fLabels.Labels()
}

func (f *Forecast) Coefficients() (map[string]float64, error) {
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

func (f *Forecast) Intercept() float64 {
	return f.intercept
}

func (f *Forecast) Model() Model {
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
		Options: f.opt,
		Weights: w,
	}
	return m
}

func (f *Forecast) ModelEq() (string, error) {
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

func (f *Forecast) Scores() Scores {
	if f.scores == nil {
		return Scores{}
	}
	return *f.scores
}

func (f *Forecast) Residuals() []float64 {
	res := make([]float64, len(f.residual))
	copy(res, f.residual)
	return res
}

func (f *Forecast) TrendComponent() []float64 {
	res := make([]float64, len(f.trend))
	copy(res, f.trend)
	return res
}

func (f *Forecast) SeasonalityComponent() []float64 {
	res := make([]float64, len(f.seasonality))
	copy(res, f.seasonality)
	return res
}
