package forecast

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/aouyang1/go-forecast/timedataset"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

var (
	ErrNoTrainingData           = errors.New("no training data")
	ErrLabelExists              = errors.New("label already exists in TimeDataset")
	ErrMismatchedDataLen        = errors.New("input data has different length than time")
	ErrFeatureLabelsInitialized = errors.New("feature labels already initialized")
	ErrUnknownTimeFeature       = errors.New("unknown time feature")
	ErrNoModelCoefficients      = errors.New("no model coefficients from fit")
)

type Forecast struct {
	opt    *Options
	scores *Scores // score calculations after training

	// model coefficients
	fLabels   []string // index positions correspond to coefficient values
	residual  []float64
	coef      []float64
	intercept float64
}

func New(opt *Options) (*Forecast, error) {
	if opt == nil {
		opt = NewDefaultOptions()
	}

	return &Forecast{opt: opt}, nil
}

func (f *Forecast) generateFeatures(t []time.Time) (map[string][]float64, error) {
	tFeat := generateTimeFeatures(t, f.opt)

	feat, err := generateFourierFeatures(tFeat, f.opt)
	if err != nil {
		return nil, err
	}

	// do not include weekly fourier features if time range is less than 1 week
	if t[len(t)-1].Sub(t[0]) < time.Duration(7*24*time.Hour) {
		for label := range feat {
			if strings.HasPrefix(label, "dow") {
				delete(feat, label)
			}
		}
	}
	return feat, nil
}

func (f *Forecast) Fit(trainingData *timedataset.TimeDataset) error {
	if trainingData == nil {
		return ErrNoTrainingData
	}

	// generate features
	x, err := f.generateFeatures(trainingData.T)
	if err != nil {
		return err
	}

	// prune linearly dependent fourier components
	f.fLabels = featureLabels(x)
	features := featureMatrix(trainingData.T, f.fLabels, x)
	observations := observationMatrix(trainingData.Y)
	f.intercept, f.coef = OLS(features, observations)

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

	return nil
}

func (f *Forecast) Predict(t []time.Time) ([]float64, error) {
	// generate features
	x, err := f.generateFeatures(t)
	if err != nil {
		return nil, err
	}

	// prune linearly dependent fourier components
	f.fLabels = featureLabels(x)
	features := featureMatrix(t, f.fLabels, x).T()
	weights := []float64{f.intercept}
	weights = append(weights, f.coef...)
	w := mat.NewDense(1, len(f.fLabels)+1, weights)

	var resMx mat.Dense
	resMx.Mul(w, features)

	return mat.Row(nil, 0, &resMx), nil
}

func (f *Forecast) FeatureLabels() []string {
	dst := make([]string, len(f.fLabels))
	copy(dst, f.fLabels)
	return dst
}

func (f *Forecast) Coefficients() (map[string]float64, error) {
	labels := f.fLabels
	if len(labels) == 0 || len(f.coef) == 0 {
		return nil, ErrNoModelCoefficients
	}
	coef := make(map[string]float64)
	for i := 0; i < len(f.coef); i++ {
		coef[labels[i]] = f.coef[i]
	}
	return coef, nil
}

func (f *Forecast) Intercept() float64 {
	return f.intercept
}

func (f *Forecast) ModelEq() (string, error) {
	eq := "y ~ "

	coef, err := f.Coefficients()
	if err != nil {
		return "", err
	}

	eq += fmt.Sprintf("%.2f", f.Intercept())
	labels := f.fLabels
	for i := 0; i < len(f.coef); i++ {
		eq += fmt.Sprintf("+%.2f*%s", coef[labels[i]], labels[i])
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
